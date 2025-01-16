import numpy as np
import warnings
import logging
warnings.filterwarnings('ignore')

# Configure logging at the top of your slicer.py
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# This function is obtained from librosa.
def get_rms(
    y,
    *,
    frame_length=2048,
    hop_length=512,
    pad_mode="constant",
):
    padding = (int(frame_length // 2), int(frame_length // 2))
    y = np.pad(y, padding, mode=pad_mode)

    axis = -1
    # put our new within-frame axis at the end for now
    out_strides = y.strides + tuple([y.strides[axis]])
    # Reduce the shape on the framing axis
    x_shape_trimmed = list(y.shape)
    x_shape_trimmed[axis] -= frame_length - 1
    out_shape = tuple(x_shape_trimmed) + tuple([frame_length])
    xw = np.lib.stride_tricks.as_strided(
        y, shape=out_shape, strides=out_strides
    )
    if axis < 0:
        target_axis = axis - 1
    else:
        target_axis = axis + 1
    xw = np.moveaxis(xw, -1, target_axis)
    # Downsample along the target axis
    slices = [slice(None)] * xw.ndim
    slices[axis] = slice(0, None, hop_length)
    x = xw[tuple(slices)]

    # Calculate power
    power = np.mean(np.abs(x) ** 2, axis=-2, keepdims=True)

    return np.sqrt(power)


class Slicer:
    def __init__(self,
                 sr: int,
                 threshold: float = -30.,
                 min_length: int = 3000,
                 min_interval: int = 100,
                 hop_size: int = 10,
                 max_sil_kept: int = 300,
                 look_ahead_frames: int = 4):
        if not min_length >= min_interval >= hop_size:
            raise ValueError('The following condition must be satisfied: min_length >= min_interval >= hop_size')
        if not max_sil_kept >= hop_size:
            raise ValueError('The following condition must be satisfied: max_sil_kept >= hop_size')
        min_interval = sr * min_interval / 1000
        self.sr = sr
        self.threshold = 10 ** (threshold / 20.)
        self.hop_size = round(sr * hop_size / 1000)
        self.win_size = min(round(min_interval), 4 * self.hop_size)
        self.min_length = round(sr * min_length / 1000 / self.hop_size)
        self.min_interval = round(min_interval / self.hop_size)
        self.max_sil_kept = round(sr * max_sil_kept / 1000 / self.hop_size)
        self.look_ahead = look_ahead_frames

    def _find_zero_crossing(self, waveform, start_idx, end_idx, direction='forward'):
        """Find the nearest zero crossing point in the given range."""
        if len(waveform.shape) > 1:
            samples = waveform.mean(axis=0)
        else:
            samples = waveform
            
        # Ensure we stay within bounds
        start_idx = max(0, start_idx)
        end_idx = min(len(samples), end_idx)
        
        # Convert indices to time (in seconds)
        start_time = start_idx / self.sr
        end_time = end_idx / self.sr
        
        logger.debug(f"_find_zero_crossing called with start_idx={start_idx} ({start_time:.3f}s), end_idx={end_idx} ({end_time:.3f}s), direction={direction}")

        if direction == 'forward':
            search_range = range(start_idx, end_idx - 1)
        else:  # backward
            search_range = range(end_idx - 2, start_idx - 1, -1)
            
        for i in search_range:
            if samples[i] * samples[i + 1] <= 0:  # Zero crossing found
                # Determine which point is closer to zero
                closer_point = i if abs(samples[i]) < abs(samples[i + 1]) else i + 1
                logger.debug(f"Zero crossing found at index {closer_point}")
                return closer_point

        logger.debug("No zero crossing found in the specified range.")
        return start_idx if direction == 'forward' else end_idx - 1

    def _find_best_cut_point(self, waveform, frame_idx, is_start=False):
        """Find the best cut point near the given frame index."""
        # Convert frame index to sample index
        sample_idx = frame_idx * self.hop_size
        
        # For the start of audio, we want a clean ramp up from true silence
        if is_start:
            # Look for the first non-zero sample
            if len(waveform.shape) > 1:
                samples = waveform.mean(axis=0)
            else:
                samples = waveform
            
            # Define search window
            search_end = min(len(samples), sample_idx + self.hop_size * 2)
            
            # Find first significant sample (above noise floor)
            noise_floor = self.threshold * 0.1  # More sensitive threshold for start detection
            for i in range(sample_idx, search_end):
                if abs(samples[i]) > noise_floor:
                    # Back up a few samples to ensure clean start
                    return max(0, i - 32) // self.hop_size  # 32 samples for small padding
            
            return sample_idx // self.hop_size
            
        # Normal zero-crossing search for other positions
        window_size = self.hop_size
        start_search = max(0, sample_idx - window_size)
        end_search = min(len(waveform) if len(waveform.shape) == 1 else waveform.shape[1], 
                        sample_idx + window_size)
        
        cut_point = self._find_zero_crossing(waveform, start_search, end_search)
        return cut_point // self.hop_size

    def _apply_slice(self, waveform, begin, end):
        """Apply slice with zero-crossing adjustment."""
        # Find actual cut points at zero crossings
        actual_begin = self._find_zero_crossing(waveform, 
                                              begin * self.hop_size, 
                                              (begin + self.look_ahead) * self.hop_size,
                                              'forward')
        actual_end = self._find_zero_crossing(waveform,
                                            (end - self.look_ahead) * self.hop_size,
                                            end * self.hop_size,
                                            'backward')
        
        if len(waveform.shape) > 1:
            return waveform[:, actual_begin:actual_end]
        else:
            return waveform[actual_begin:actual_end]

    def slice(self, waveform):
        if len(waveform.shape) > 1:
            samples = waveform.mean(axis=0)
        else:
            samples = waveform
        if (samples.shape[0] + self.hop_size - 1) // self.hop_size <= self.min_length:
            # Find optimal start point even for single-chunk case
            start_pos = self._find_best_cut_point(waveform, 0, is_start=True) * self.hop_size
            return [(start_pos, waveform[start_pos:])]

        rms_list = get_rms(y=samples, frame_length=self.win_size, hop_length=self.hop_size).squeeze(0)
        sil_tags = []
        silence_start = None
        clip_start = 0
        
        # Detect silence regions
        for i, rms in enumerate(rms_list):
            # Keep looping while frame is silent
            if rms < self.threshold:
                # Record start of silent frames
                if silence_start is None:
                    silence_start = i
                continue
            # Keep looping while frame is not silent and silence start has not been recorded
            if silence_start is None:
                continue
            # Clear recorded silence start if interval is not enough or clip is too short
            is_leading_silence = silence_start == 0 and i > self.max_sil_kept
            need_slice_middle = i - silence_start >= self.min_interval and i - clip_start >= self.min_length
            if not is_leading_silence and not need_slice_middle:
                silence_start = None
                continue
            # Need slicing. Record the range of silent frames to be removed
            if i - silence_start <= self.max_sil_kept:
                pos = rms_list[silence_start: i + 1].argmin() + silence_start
                if silence_start == 0:
                    sil_tags.append((0, pos))
                else:
                    sil_tags.append((pos, pos))
                clip_start = pos
            elif i - silence_start <= self.max_sil_kept * 2:
                pos = rms_list[i - self.max_sil_kept: silence_start + self.max_sil_kept + 1].argmin()
                pos += i - self.max_sil_kept
                pos_l = rms_list[silence_start: silence_start + self.max_sil_kept + 1].argmin() + silence_start
                pos_r = rms_list[i - self.max_sil_kept: i + 1].argmin() + i - self.max_sil_kept
                if silence_start == 0:
                    sil_tags.append((0, pos_r))
                    clip_start = pos_r
                else:
                    sil_tags.append((min(pos_l, pos), max(pos_r, pos)))
                    clip_start = max(pos_r, pos)
            else:
                pos_l = rms_list[silence_start: silence_start + self.max_sil_kept + 1].argmin() + silence_start
                pos_r = rms_list[i - self.max_sil_kept: i + 1].argmin() + i - self.max_sil_kept
                if silence_start == 0:
                    sil_tags.append((0, pos_r))
                else:
                    sil_tags.append((pos_l, pos_r))
                clip_start = pos_r
            silence_start = None

        # Deal with trailing silence
        total_frames = rms_list.shape[0]
        if silence_start is not None and total_frames - silence_start >= self.min_interval:
            silence_end = min(total_frames, silence_start + self.max_sil_kept)
            pos = rms_list[silence_start: silence_end + 1].argmin() + silence_start
            sil_tags.append((pos, total_frames + 1))

        # Apply and return slices
        if len(sil_tags) == 0:
            # Find optimal start point
            start_pos = self._find_best_cut_point(waveform, 0, is_start=True) * self.hop_size
            return [(start_pos, waveform[start_pos:])]
        else:
            chunks_with_pos = []
            if sil_tags[0][0] > 0:
                # Find optimal starting point for first chunk
                start_frame = self._find_best_cut_point(waveform, 0, is_start=True)
                end_frame = self._find_best_cut_point(waveform, sil_tags[0][0])
                start_pos = start_frame * self.hop_size
                chunks_with_pos.append((
                    start_pos,
                    self._apply_slice(waveform, start_frame, end_frame)
                ))

            for i in range(len(sil_tags) - 1):
                start_frame = self._find_best_cut_point(waveform, sil_tags[i][1])
                end_frame = self._find_best_cut_point(waveform, sil_tags[i + 1][0])
                start_pos = start_frame * self.hop_size
                chunks_with_pos.append((
                    start_pos,
                    self._apply_slice(waveform, start_frame, end_frame)
                ))

            if sil_tags[-1][1] < total_frames:
                start_frame = self._find_best_cut_point(waveform, sil_tags[-1][1])
                start_pos = start_frame * self.hop_size
                chunks_with_pos.append((
                    start_pos,
                    self._apply_slice(waveform, start_frame, total_frames)
                ))

            return chunks_with_pos


def main():
    import os.path
    from argparse import ArgumentParser
    import librosa
    import soundfile

    parser = ArgumentParser()
    parser.add_argument('audio', type=str, help='The audio to be sliced')
    parser.add_argument('--out', type=str, help='Output directory of the sliced audio clips')
    parser.add_argument('--db_thresh', type=float, required=False, default=-30,
                        help='The dB threshold for silence detection')
    parser.add_argument('--min_length', type=int, required=False, default=3000,
                        help='The minimum milliseconds required for each sliced audio clip')
    parser.add_argument('--min_interval', type=int, required=False, default=100,
                        help='The minimum milliseconds for a silence part to be sliced')
    parser.add_argument('--hop_size', type=int, required=False, default=10,
                        help='Frame length in milliseconds')
    parser.add_argument('--max_sil_kept', type=int, required=False, default=300,
                        help='The maximum silence length kept around the sliced clip, presented in milliseconds')
    args = parser.parse_args()
    
    out = args.out
    if out is None:
        out = os.path.dirname(os.path.abspath(args.audio))
    
    audio, sr = librosa.load(args.audio, sr=None, mono=False)
    slicer = Slicer(
        sr=sr,
        threshold=args.db_thresh,
        min_length=args.min_length,
        min_interval=args.min_interval,
        hop_size=args.hop_size,
        max_sil_kept=args.max_sil_kept
    )
    
    chunks_with_pos = slicer.slice(audio)
    
    if not os.path.exists(out):
        os.makedirs(out)
    
    for i, (pos, chunk) in enumerate(chunks_with_pos):
        if len(chunk.shape) > 1:
            chunk = chunk.T
        soundfile.write(
            os.path.join(out, f'%s_%d_pos_%d.wav' % (
                os.path.basename(args.audio).rsplit('.', maxsplit=1)[0], 
                i,
                pos
            )), 
            chunk, 
            sr
        )


if __name__ == '__main__':
    main()
