"""
Jump counting core logic module

Implements the state machine-based jump detection and counting algorithm.
Uses binary pattern matching to identify stable jump events and filter noise.
"""


class JumpCounter:
    """Core jump counting logic using binary state machine

    This class implements a sophisticated jump detection algorithm that uses
    a sliding window of binary predictions to identify stable jump events.
    The algorithm filters out noise and false positives by requiring specific
    binary patterns before counting a jump.

    The state machine tracks the last 4 prediction bits and looks for patterns:
    - Pattern 7 (0111): Rising edge detection - triggers jump count
    - Pattern 15 (1111): Sustained rising state
    - Other patterns: Non-rising or unstable states
    """

    def __init__(self):
        """Initialize the jump counter

        Sets up the counter state with zero jumps and empty binary history.
        """
        self.jump_cnt = 0
        self.jump_cnt_binary_mark = 0  # Binary history buffer (4 bits)

    def process_prediction(self, prob: float, threshold: float) -> tuple[bool, int]:
        """Process model prediction and update jump count

        Takes the model's jump probability prediction, applies threshold,
        and updates the internal state machine. Returns whether the person
        is currently in a rising motion and the total jump count.

        Args:
            prob: Model prediction probability (0.0-1.0)
            threshold: Decision threshold for binary classification

        Returns:
            tuple: (is_rising, total_jump_count)
                - is_rising: True if person is currently jumping up
                - total_jump_count: Total number of jumps detected
        """
        # Convert probability to binary prediction
        y_pred = int((prob > threshold))

        # Update 4-bit sliding window of predictions
        mark1 = (self.jump_cnt_binary_mark << 1) & 0b1111
        self.jump_cnt_binary_mark = (mark1 | y_pred) & 0b1111

        # State machine logic for jump detection
        if self.jump_cnt_binary_mark in [7, 15]:  # 0111 or 1111 patterns
            is_on_rising = True
            if self.jump_cnt_binary_mark == 7:  # Rising edge (0111)
                self.jump_cnt += 1  # Count new jump
        else:  # Other patterns: 0000, 0001, 0010, 0100, 0101, 0110, etc.
            is_on_rising = False

        return is_on_rising, self.jump_cnt

    def get_count(self) -> int:
        """Get current jump count

        Returns:
            int: Total number of jumps detected since initialization or last reset
        """
        return self.jump_cnt

    def reset(self):
        """Reset the counter to initial state

        Clears jump count and binary history. Useful for starting a new
        counting session or recovering from errors.
        """
        self.jump_cnt = 0
        self.jump_cnt_binary_mark = 0
