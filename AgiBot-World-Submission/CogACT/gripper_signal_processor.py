import numpy as np

class GripperSignalFilter:
    def __init__(
        self,
        ema_alpha=0.25,          # smoothing (0.15–0.35 is typical)
        max_step=0.10,           # max change per step (rate limit)
        dropout_abs=0.05,        # treat near-0 as a dropout candidate
        dropout_rel_drop=0.4,    # sudden relative drop vs previous
        lookahead=2,             # frames to confirm quick recovery
        monotone=None            # 'closing' (non-decreasing) or 'opening' (non-increasing) or None
    ):
        self.ema_alpha = ema_alpha
        self.max_step = max_step
        self.dropout_abs = dropout_abs
        self.dropout_rel_drop = dropout_rel_drop
        self.lookahead = lookahead
        self.monotone = monotone
        self._ema = None
        self._hist = []  # keep a tiny buffer for dropout repair

    def _repair_dropout(self, x_t):
        """
        Detect & repair one-frame dropouts:
        if x drops hard vs last value and the very recent history suggests recovery,
        replace by linear interpolation between last and recovery.
        """
        self._hist.append(x_t)
        if len(self._hist) < 3:
            return x_t

        # previous smoothed (or raw last) value
        x_prev = self._hist[-2]
        x_curr = self._hist[-1]

        sudden_drop = (x_prev - x_curr) > self.dropout_rel_drop
        near_zero = x_curr <= self.dropout_abs

        if sudden_drop or near_zero:
            # if we have a recovery in the next <= lookahead frames (if available), interpolate
            # (online version: use what we already have; for streaming, this acts as a one-step lag)
            for k in range(1, min(self.lookahead + 1, len(self._hist) - 1)):
                x_future = self._hist[-1 - k]  # look backwards since we only have past
                # If the recent past (before the drop) is much higher, we consider this a glitch
                if x_future >= x_prev - self.dropout_rel_drop / 2:
                    # Interpolate between x_future and x_prev across k steps
                    repaired = x_prev  # simplest: hold last good value
                    self._hist[-1] = repaired
                    return repaired

            # If no clear recovery info, still clamp away from near-zero glitch by holding last
            if near_zero:
                self._hist[-1] = x_prev
                return x_prev

        return x_t

    def step(self, x_t):
        # 1) clamp
        x_t = float(np.clip(x_t, 0.0, 1.0))

        # 2) dropout repair (single-frame 0s or big sudden dips)
        x_t = self._repair_dropout(x_t)

        # 3) EMA smoothing
        if self._ema is None:
            self._ema = x_t
        else:
            self._ema = self.ema_alpha * x_t + (1 - self.ema_alpha) * self._ema

        y = self._ema

        # 4) optional monotone constraint if you *know* the phase for this segment
        if self.monotone == "closing":
            y = max(y, self._hist[-2] if len(self._hist) >= 2 else y)
        elif self.monotone == "opening":
            y = min(y, self._hist[-2] if len(self._hist) >= 2 else y)

        # 5) rate limit (physically plausible change per tick)
        if len(self._hist) >= 2:
            y_prev = self._hist[-2]
            dy = np.clip(y - y_prev, -self.max_step, self.max_step)
            y = y_prev + dy

        # final clamp
        y = float(np.clip(y, 0.0, 1.0))
        # store the filtered value as the “last”
        self._hist[-1] = y
        return y

if __name__ == "__main__":
    # Example usage
    filter = GripperSignalFilter()
    # raw_signals = [0.1, 0.2,0, 0.05, 0.3, 0.25, 0, 0]
    raw_signals = [0.1, 0.2,0, 0.05, 0.3, 0.25, 0.25, 0.25]
    for signal in raw_signals:
        filtered_signal = filter.step(signal)
        print(f"Raw: {signal}, Filtered: {filtered_signal}")

    flt = GripperSignalFilter(
        ema_alpha=0.25,
        max_step=0.08,          # tune to your control rate & gripper speed
        dropout_abs=0.05,
        dropout_rel_drop=0.5,   # “glitch” size
        lookahead=1,
        monotone=None           # or 'closing'/'opening' inside known phases
        )

    filtered = []
    for v in raw_signals:        # raw_signal values in [0,1]
        filtered.append(flt.step(v))

    print(f"Raw: {v}, Filtered: {filtered[-1]}")

    # Respect the Max by find max of raw and filter, and apply to filtered
    max_raw = max(raw_signals)
    max_filtered = max(filtered)
    if max_raw > 0:
        filtered = [f * (max_raw / max_filtered) for f in filtered]
    print(f"Final Filtered: {filtered}")
