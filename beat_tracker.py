import os, sys
import numpy as np
import librosa
import scipy.signal
from scipy.stats import mode
import matplotlib.pyplot as plt
import librosa
import librosa.display


import mir_eval

FRAME_SIZE = 2048
HOP_SIZE = 512
COMPRESSION_CONSTANT = 1000.0
DISTANCE_SEC = 0.3
TOLERANCE = 0.07


def rms_odf(snd, sr, hop_size=HOP_SIZE):
    num_frames = snd.shape[1]
    odf = np.zeros(num_frames)
    for i in range(num_frames):
        mag = np.abs(snd[:, i])
        odf[i] = np.sqrt(np.mean(mag**2))
    if np.max(odf) > 0:
        odf /= np.max(odf)
    fps = sr / hop_size
    return odf, fps


def hfc_odf(snd, sr, hop_size=HOP_SIZE):
    freq_bins, num_frames = snd.shape
    odf = np.zeros(num_frames)
    weights = np.arange(1, freq_bins + 1)
    for i in range(num_frames):
        mag_sq = np.abs(snd[:, i]) ** 2
        odf[i] = np.sum(mag_sq * weights)
    if np.max(odf) > 0:
        odf /= np.max(odf)
    fps = sr / hop_size
    return odf, fps


def sf_odf(snd, sr, hop_size=HOP_SIZE):
    freq_bins, num_frames = snd.shape
    odf = np.zeros(num_frames)
    prev_mag = np.zeros(freq_bins)
    for i in range(num_frames):
        mag = np.abs(snd[:, i])
        diff = mag - prev_mag
        odf[i] = np.sum(diff[diff > 0])
        prev_mag = mag
    if np.max(odf) > 0:
        odf /= np.max(odf)
    fps = sr / hop_size
    return odf, fps


def superflux_odf(snd, sr, hop_size=HOP_SIZE, c=10.0, freq_filt_len=3):
    freq_bins, num_frames = snd.shape
    odf = np.zeros(num_frames)
    prev_localmax = None
    for i in range(num_frames):
        mag = np.abs(snd[:, i])
        log_mag = np.log1p(c * mag)
        if freq_filt_len == 3:
            localmax = np.maximum.reduce([log_mag[:-2], log_mag[1:-1], log_mag[2:]])
        elif freq_filt_len == 5:
            localmax = np.maximum.reduce(
                [log_mag[:-4], log_mag[1:-3], log_mag[2:-2], log_mag[3:-1], log_mag[4:]]
            )
        else:
            localmax = np.maximum.reduce([log_mag[:-2], log_mag[1:-1], log_mag[2:]])
        if i > 0:
            diff = localmax - prev_localmax
            odf[i] = np.sum(diff[diff > 0])
        prev_localmax = localmax
    if np.max(odf) > 0:
        odf /= np.max(odf)
    fps = sr / hop_size
    return odf, fps


def cd_odf(snd, sr, hop_size=HOP_SIZE):
    freq_bins, num_frames = snd.shape
    odf = np.zeros(num_frames)
    prev_mag = np.zeros(freq_bins)
    prev_phase = np.zeros(freq_bins)
    prevprev_phase = np.zeros(freq_bins)
    for i in range(num_frames):
        mag = np.abs(snd[:, i])
        phase = np.angle(snd[:, i])
        if i < 2:
            odf[i] = 0.0
        else:
            tPhase = 2 * prev_phase - prevprev_phase
            cdTerm = (prev_mag**2 + mag**2) - 2 * prev_mag * mag * np.cos(
                phase - tPhase
            )
            cdTerm = np.maximum(cdTerm, 0)
            cdVector = np.sqrt(cdTerm)
            odf[i] = np.mean(cdVector)
        prevprev_phase = prev_phase
        prev_phase = phase
        prev_mag = mag
    if np.max(odf) > 0:
        odf /= np.max(odf)
    fps = sr / hop_size
    return odf, fps


def rcd_odf(snd, sr, hop_size=HOP_SIZE):
    freq_bins, num_frames = snd.shape
    odf = np.zeros(num_frames)
    prev_mag = np.zeros(freq_bins)
    prev_phase = np.zeros(freq_bins)
    prevprev_phase = np.zeros(freq_bins)
    for i in range(num_frames):
        mag = np.abs(snd[:, i])
        phase = np.angle(snd[:, i])
        if i < 2:
            odf[i] = 0.0
        else:
            tPhase = 2 * prev_phase - prevprev_phase
            cdTerm = (prev_mag**2 + mag**2) - 2 * prev_mag * mag * np.cos(
                phase - tPhase
            )
            cdTerm = np.maximum(cdTerm, 0)
            cdVector = np.sqrt(cdTerm)
            mask = mag >= prev_mag
            odf[i] = np.mean(cdVector[mask]) if np.any(mask) else 0.0
        prevprev_phase = prev_phase
        prev_phase = phase
        prev_mag = mag
    if np.max(odf) > 0:
        odf /= np.max(odf)
    fps = sr / hop_size
    return odf, fps


def pd_odf(snd, sr, hop_size=HOP_SIZE):
    freq_bins, num_frames = snd.shape
    odf = np.zeros(num_frames)
    prev_phase = np.zeros(freq_bins)
    prevprev_phase = np.zeros(freq_bins)
    for i in range(num_frames):
        phase = np.angle(snd[:, i])
        if i < 2:
            odf[i] = 0.0
        else:
            tPhase = 2 * prev_phase - prevprev_phase
            tPhase = np.nan_to_num(tPhase)
            pdVector = np.abs(((phase - tPhase + np.pi) % (2 * np.pi)) - np.pi) / np.pi
            odf[i] = np.mean(pdVector)
        prevprev_phase = prev_phase
        prev_phase = phase
    if np.max(odf) > 0:
        odf /= np.max(odf)
    fps = sr / hop_size
    return odf, fps


def wpd_odf(snd, sr, hop_size=HOP_SIZE):
    freq_bins, num_frames = snd.shape
    odf = np.zeros(num_frames)
    prev_mag = np.zeros(freq_bins)
    prev_phase = np.zeros(freq_bins)
    prevprev_phase = np.zeros(freq_bins)
    for i in range(num_frames):
        mag = np.abs(snd[:, i])
        phase = np.angle(snd[:, i])
        rms = np.sqrt(np.mean(mag**2))
        if i < 2:
            odf[i] = 0.0
        else:
            tPhase = 2 * prev_phase - prevprev_phase
            tPhase = np.nan_to_num(tPhase)
            pdVector = np.abs(((phase - tPhase + np.pi) % (2 * np.pi)) - np.pi) / np.pi
            pdVector = np.nan_to_num(pdVector)
            if rms > 0:
                odf[i] = np.mean(pdVector * mag) / (2 * rms)
            else:
                odf[i] = 0.0
        prevprev_phase = prev_phase
        prev_phase = phase
        prev_mag = mag
    if np.max(odf) > 0:
        odf /= np.max(odf)
    fps = sr / hop_size
    return odf, fps


def sflf_odf(snd, sr, hop_size=HOP_SIZE):
    S = np.abs(snd)
    log_S = np.log1p(COMPRESSION_CONSTANT * S)
    novelty = np.sum(np.diff(log_S, axis=1).clip(min=0), axis=0)
    fps = sr / hop_size
    return novelty, fps


def energy_flux(snd, sr, hop_size=HOP_SIZE):
    energy = np.sum(np.abs(snd) ** 2, axis=0)
    novelty = np.diff(energy).clip(min=0)
    fps = sr / hop_size
    return novelty, fps


def normalise(nov):
    nov = nov - np.min(nov)
    return nov / np.max(nov) if np.max(nov) > 0 else nov


def detect_beats(novelty, fps, alpha=0.7, min_beat_interval=DISTANCE_SEC):
    novelty = normalise(novelty.flatten())
    threshold = alpha * np.std(novelty)
    min_distance_frames = int(min_beat_interval * fps)
    peaks, _ = scipy.signal.find_peaks(
        novelty, height=threshold, distance=min_distance_frames
    )
    beat_times = peaks / fps
    return beat_times


def beat_pipeline(novelty_func, y, sr, frame_size=FRAME_SIZE, hop_size=HOP_SIZE):
    snd = librosa.stft(y=y, n_fft=frame_size, hop_length=hop_size, window="hann")
    novelty, fps = novelty_func(snd, sr, hop_size)
    novelty = normalise(novelty)
    beats = detect_beats(novelty, fps)
    return beats


def mutual_agreement(seq_i, seq_j, tolerance=TOLERANCE):
    agreement = sum(np.any(np.abs(seq_j - beat) <= tolerance) for beat in seq_i)
    return agreement / len(seq_i) if len(seq_i) > 0 else 0


def fuse_multiple_beats(seqs, tolerance=TOLERANCE):
    if len(seqs) == 0:
        return np.array([])
    total_length = len(seqs)
    agreement_scores = np.zeros(total_length)
    for i in range(total_length):
        agreement_sum = sum(
            mutual_agreement(seqs[i], seqs[j], tolerance)
            for j in range(total_length)
            if i != j
        )
        agreement_scores[i] = agreement_sum / (total_length - 1)
    top_seq = np.argmax(agreement_scores)
    second_best_seq = np.argsort(agreement_scores)[-2]
    merged_beats = np.sort(np.concatenate([seqs[top_seq], seqs[second_best_seq]]))
    fused_beats = []
    cluster = [merged_beats[0]]
    for beat in merged_beats[1:]:
        if beat - cluster[-1] <= tolerance:
            cluster.append(beat)
        else:
            fused_beats.append(np.mean(cluster))
            cluster = [beat]

    return np.array(fused_beats)


def estimate_downbeats(beats):
    intervals = np.diff(beats)
    hist, bin_edges = np.histogram(intervals, bins=20)
    dominant_interval = bin_edges[np.argmax(hist)]
    beat_grouping = np.round(intervals / dominant_interval)
    common_grouping = mode(beat_grouping, keepdims=False).mode
    if common_grouping == 3:
        downbeats = beats[::3]
    else:
        downbeats = beats[::4]
    return np.array(downbeats)


def load_ground_truth_beats(annotation_file):
    beats = []
    downbeats = []

    with open(annotation_file, "r") as f:
        for line in f:
            tokens = line.strip().split()
            if tokens:
                try:
                    time = float(tokens[0])
                    label = int(tokens[1])
                    beats.append(time)
                    if label == 1:
                        downbeats.append(time)
                except ValueError:
                    continue

    return np.array(beats), np.array(downbeats)


def plot_results(y, sr, beat_times, gt_beats, filename):
    print("[Checkpoint] Plotting waveform with detected beats...")
    plt.figure(figsize=(14, 6))
    librosa.display.waveshow(y, sr=sr, alpha=0.5, color="blue")
    plt.vlines(
        beat_times, ymin=-1, ymax=1, color="red", linestyle="-", label="Predicted Beats"
    )
    plt.vlines(
        gt_beats,
        ymin=-1,
        ymax=1,
        color="green",
        linestyle="--",
        label="Ground Truth Beats",
    )
    plt.title("Waveform with Predicted Beats")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.savefig(filename)
    plt.show()
    print("[Checkpoint] Plot displayed.")


def beatTracker(inputFile, annotation_file="", plot=False):
    snd, sr = librosa.load(inputFile, sr=None)
    onset_methods = [
        ("rms", rms_odf),
        ("hfc", hfc_odf),
        ("sf", sf_odf),
        ("sflux", superflux_odf),
        ("c_diff", cd_odf),
        ("rcd", rcd_odf),
        ("phase_dev", pd_odf),
        ("w_phase_dev", wpd_odf),
        ("energy_flux", energy_flux),
        ("sflf", sflf_odf),
    ]

    beat_sequences = []
    for _, func in onset_methods:
        beats = beat_pipeline(func, snd, sr)
        beat_sequences.append(beats)
    fused_beats = fuse_multiple_beats(beat_sequences)
    downbeats = estimate_downbeats(fused_beats)

    scores = 0
    db_scores = 0
    if annotation_file:
        if not anno_file.lower().endswith(".beats"):
            print("Not a valid annotation file")
            sys.exit(1)
        gt_beats, gt_downbeats = load_ground_truth_beats(annotation_file)
        scores = mir_eval.beat.f_measure(gt_beats, fused_beats)
        db_scores = mir_eval.beat.f_measure(gt_downbeats, downbeats)

    if plot and annotation_file:
        plot_results(snd, sr, fused_beats, gt_beats, "performancee.png")

    return fused_beats, downbeats, scores, db_scores


if __name__ == "__main__":
    audio_file = "BallroomData/Quickstep/Albums-AnaBelen_Veneo-11.wav"
    anno_file = ""

    if audio_file:
        if not audio_file.lower().endswith(".wav"):
            print("Not a valid audio type")
            sys.exit(1)
        beats, downbeats, scores, db_scores = beatTracker(audio_file, anno_file, True)
        print("Beat Times:", beats)
        print("Downbeat Times:", downbeats)
        if scores:
            print("beat scores :", scores)
        if db_scores:
            print("Downbeat scores :", db_scores)
    else:
        print("No Audio file loaded")
