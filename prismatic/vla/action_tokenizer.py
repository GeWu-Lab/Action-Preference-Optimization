"""
action_tokenizer.py

Extension class that wraps base LLM/VLM tokenizer with logic to discretize and tokenize continuous robot actions.
Converts continuous action values into discrete tokens using uniform binning strategy.
"""

from typing import List, Union, Optional
import logging

import numpy as np
from transformers import PreTrainedTokenizerBase


logger = logging.getLogger(__name__)


class ActionTokenizer:
    """
    Discretizes continuous robot actions into N bins per dimension and maps them to vocabulary tokens.
    
    This tokenizer assumes a BPE-style tokenizer (like LlamaTokenizer) where the least used tokens
    appear at the end of the vocabulary. It uses the final `n_bins` tokens for action representation.
    
    Attributes:
        tokenizer: Base LLM/VLM tokenizer to extend
        n_bins: Number of bins for discretization
        min_action: Minimum action value for clipping
        max_action: Maximum action value for clipping
        bins: Array of bin edges for discretization
        bin_centers: Array of bin center values
        action_token_begin_idx: Starting index for action tokens in vocabulary
    """
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        bins: int = 256,
        min_action: Union[int, float] = -1,
        max_action: Union[int, float] = 1
    ) -> None:
        """
        Initialize the ActionTokenizer with specified parameters.

        Args:
            tokenizer: Base LLM/VLM tokenizer to extend
            bins: Number of bins for each continuous value (uniform binning strategy)
            min_action: Minimum action value for clipping and bin interval lower bound
            max_action: Maximum action value for clipping and bin interval upper bound
            
        Raises:
            ValueError: If bins <= 0 or min_action >= max_action or insufficient vocabulary size
        """
        # Input validation
        if bins <= 0:
            raise ValueError(f"Number of bins must be positive, got {bins}")
        if min_action >= max_action:
            raise ValueError(f"min_action ({min_action}) must be less than max_action ({max_action})")
        if tokenizer.vocab_size <= bins + 1:
            raise ValueError(f"Tokenizer vocab size ({tokenizer.vocab_size}) insufficient for {bins} action bins")
            
        self.tokenizer = tokenizer
        self.n_bins = bins
        self.min_action = float(min_action)
        self.max_action = float(max_action)

        # Create uniform bins and compute bin centers
        self.bins = np.linspace(self.min_action, self.max_action, self.n_bins)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0

        # Set action token starting index - uses the final `n_bins` tokens of vocabulary
        # Contract: Always overwrite the final `n_bins` tokens of the vocabulary
        self.action_token_begin_idx: int = self.tokenizer.vocab_size - (self.n_bins + 1)
        
        logger.debug(f"Initialized ActionTokenizer with {self.n_bins} bins, "
                    f"range [{self.min_action}, {self.max_action}], "
                    f"action tokens starting at index {self.action_token_begin_idx}")

    def __call__(self, action: np.ndarray) -> Union[str, List[str]]:
        """
        Encode continuous actions into token strings.
        
        Clips actions to valid range, discretizes them into bins, and maps to vocabulary tokens.
        Uses the last `n_bins` tokens of the vocabulary for action representation.

        Args:
            action: Continuous action values as numpy array (1D or 2D for batch)

        Returns:
            Decoded token string(s) representing the discretized action(s)
            - Single string for 1D input
            - List of strings for 2D batch input
            
        Raises:
            ValueError: If input is not a numpy array or has invalid dimensions
        """
        if not isinstance(action, np.ndarray):
            raise ValueError("Action must be a numpy array")
            
        # Clip actions to valid range
        clipped_action = np.clip(action, a_min=self.min_action, a_max=self.max_action)
        
        # Discretize actions using bins
        discretized_action = np.digitize(clipped_action, self.bins)
        
        # Map to token IDs (using vocabulary size minus discretized values)
        action_token_ids = self.tokenizer.vocab_size - discretized_action

        # Handle single element vs. batch processing
        if len(discretized_action.shape) == 1:
            return self.tokenizer.decode(action_token_ids.tolist())
        else:
            return self.tokenizer.batch_decode(action_token_ids.tolist())

    def decode_token_ids_to_actions(self, action_token_ids: np.ndarray) -> np.ndarray:
        """
        Decode action token IDs back to continuous action values.

        Maps discrete action token IDs to their corresponding continuous values using bin centers.
        
        Note: Due to discretization mechanics, digitization returns bin indices in [1, n_bins] range
        when there are only (n_bins - 1) bin intervals. We handle the edge case where the last
        possible index maps to the last bin interval.

        Args:
            action_token_ids: Array of action token IDs to decode

        Returns:
            Continuous action values as numpy array
            
        Example:
            With 256 bins: bins has 256 values, bin_centers has 255 values.
            Digitization returns indices [1, 256], we convert to [0, 255] range.
            Index 255 would cause out-of-bounds error, so we clip to [0, 254].
        """
        if not isinstance(action_token_ids, np.ndarray):
            action_token_ids = np.array(action_token_ids)
            
        # Convert token IDs back to discretized action indices
        discretized_actions = self.tokenizer.vocab_size - action_token_ids
        
        # Adjust indices and clip to valid range for bin_centers
        # Subtract 1 to convert from [1, n_bins] to [0, n_bins-1] range
        # Clip to prevent out-of-bounds access to bin_centers
        adjusted_indices = np.clip(
            discretized_actions - 1, 
            a_min=0, 
            a_max=self.bin_centers.shape[0] - 1
        )

        return self.bin_centers[adjusted_indices]

    @property
    def vocab_size(self) -> int:
        """Return the number of action tokens (bins) used by this tokenizer."""
        return self.n_bins
        
    @property
    def action_range(self) -> tuple[float, float]:
        """Return the action value range as (min_action, max_action) tuple."""
        return (self.min_action, self.max_action)
        
    def get_action_token_ids(self) -> np.ndarray:
        """
        Get all action token IDs used by this tokenizer.
        
        Returns:
            Array of token IDs corresponding to action tokens
        """
        return np.arange(self.action_token_begin_idx, self.tokenizer.vocab_size)
        
    def __repr__(self) -> str:
        """Return string representation of the ActionTokenizer."""
        return (f"ActionTokenizer(bins={self.n_bins}, "
                f"range=[{self.min_action}, {self.max_action}], "
                f"vocab_size={self.tokenizer.vocab_size})")
