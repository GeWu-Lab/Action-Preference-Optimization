 # Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
 # SPDX-License-Identifier: MIT
"""
action_tokenizer.py

Extension class that wraps base LLM/VLM tokenizer with logic to discretize and tokenize continuous robot actions.
"""

from typing import List, Union, Optional
import warnings

import numpy as np
from transformers import PreTrainedTokenizerBase


class ActionTokenizer:
    """
    Tokenizer for continuous robot actions that discretizes actions into bins and maps them to vocabulary tokens.
    
    This class extends a base tokenizer by adding functionality to convert continuous robot actions
    into discrete tokens using uniform binning. It assumes the tokenizer follows BPE-style conventions
    where the least frequently used tokens appear at the end of the vocabulary.
    """
    
    # Default configuration constants
    DEFAULT_BINS = 256
    DEFAULT_MIN_ACTION = -1.0
    DEFAULT_MAX_ACTION = 1.0
    
    def __init__(
        self, 
        tokenizer: PreTrainedTokenizerBase, 
        bins: int = DEFAULT_BINS, 
        min_action: float = DEFAULT_MIN_ACTION, 
        max_action: float = DEFAULT_MAX_ACTION
    ) -> None:
        """
        Initialize the ActionTokenizer with discretization parameters.

        Args:
            tokenizer: Base LLM/VLM tokenizer to extend
            bins: Number of bins for discretizing each continuous value (must be > 1)
            min_action: Minimum action value for clipping and bin interval lower bound
            max_action: Maximum action value for clipping and bin interval upper bound
            
        Raises:
            ValueError: If bins <= 1 or min_action >= max_action
            
        Note:
            By default, assumes a BPE-style tokenizer where the least used tokens
            appear at the end of the vocabulary. The last `bins` tokens will be
            used for action representation.
        """
        self._validate_inputs(bins, min_action, max_action, tokenizer)
        
        self.tokenizer = tokenizer
        self.n_bins = bins
        self.min_action = float(min_action)
        self.max_action = float(max_action)
        
        # Create uniform bins and compute bin centers
        self._initialize_bins()
        
        # Set action token begin index (assumes overwriting final n_bins tokens)
        self.action_token_begin_idx = self.tokenizer.vocab_size - (self.n_bins + 1)
        
    def _validate_inputs(
        self, 
        bins: int, 
        min_action: float, 
        max_action: float, 
        tokenizer: PreTrainedTokenizerBase
    ) -> None:
        """Validate initialization parameters."""
        if bins <= 1:
            raise ValueError(f"Number of bins must be > 1, got {bins}")
            
        if min_action >= max_action:
            raise ValueError(f"min_action ({min_action}) must be < max_action ({max_action})")
            
        if tokenizer.vocab_size <= bins:
            raise ValueError(
                f"Tokenizer vocab size ({tokenizer.vocab_size}) must be > bins ({bins})"
            )
            
        if bins > tokenizer.vocab_size // 2:
            warnings.warn(
                f"Using {bins} bins with vocab size {tokenizer.vocab_size} may impact "
                "tokenizer performance. Consider reducing bins or using a larger vocabulary.",
                UserWarning
            )
    
    def _initialize_bins(self) -> None:
        """Initialize uniform bins and compute bin centers."""
        self.bins = np.linspace(self.min_action, self.max_action, self.n_bins)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0
        
    def __call__(self, action: np.ndarray) -> Union[str, List[str]]:
        """
        Discretize and tokenize continuous actions.
        
        Args:
            action: Continuous action values as numpy array
            
        Returns:
            Decoded token string(s) representing the discretized actions
            
        Note:
            Actions are clipped to [min_action, max_action] range and mapped to
            the last n_bins tokens of the vocabulary.
        """
        if not isinstance(action, np.ndarray):
            action = np.asarray(action)
            
        # Clip actions to valid range
        clipped_action = np.clip(action, self.min_action, self.max_action)
        
        # Discretize actions using uniform bins
        discretized_action = np.digitize(clipped_action, self.bins)
        
        # Convert to token IDs (using vocabulary end tokens)
        token_ids = self.tokenizer.vocab_size - discretized_action
        
        # Handle single element vs batch processing
        if discretized_action.ndim == 1 and len(discretized_action.shape) == 1:
            return self.tokenizer.decode(token_ids.tolist())
        else:
            return self.tokenizer.batch_decode(token_ids.tolist())

    def decode_token_ids_to_actions(self, action_token_ids: np.ndarray) -> np.ndarray:
        """
        Convert discrete action token IDs back to continuous actions.

        Args:
            action_token_ids: Token IDs representing discretized actions
            
        Returns:
            Continuous action values corresponding to bin centers
            
        Note:
            Due to discretization mechanics, bin indices are in range [1, n_bins].
            We map these to bin centers in range [0, n_bins-2] to avoid out-of-bounds
            errors. The last bin index is clamped to the final bin center.
            
        Example:
            With 256 bins, we have 255 bin centers. Digitization returns indices
            [1, 256], which we convert to [0, 255] then clamp to [0, 254] to
            safely index into bin_centers array.
        """
        if not isinstance(action_token_ids, np.ndarray):
            action_token_ids = np.asarray(action_token_ids)
            
        # Convert token IDs back to discretized actions
        discretized_actions = self.tokenizer.vocab_size - action_token_ids
        
        # Adjust indices for bin center lookup (handle edge case)
        bin_indices = np.clip(
            discretized_actions - 1, 
            a_min=0, 
            a_max=len(self.bin_centers) - 1
        )
        
        return self.bin_centers[bin_indices]

    @property
    def vocab_size(self) -> int:
        """Return the number of action tokens (bins) used by this tokenizer."""
        return self.n_bins
        
    @property
    def action_range(self) -> tuple[float, float]:
        """Return the valid action range as (min_action, max_action)."""
        return (self.min_action, self.max_action)
        
    def get_action_token_ids(self) -> List[int]:
        """
        Get the list of token IDs used for action representation.
        
        Returns:
            List of token IDs from the end of vocabulary used for actions
        """
        return list(range(
            self.action_token_begin_idx, 
            self.tokenizer.vocab_size
        ))
        
    def __repr__(self) -> str:
        """Return string representation of the ActionTokenizer."""
        return (
            f"ActionTokenizer(bins={self.n_bins}, "
            f"range=[{self.min_action}, {self.max_action}], "
            f"vocab_size={self.tokenizer.vocab_size})"
        )
