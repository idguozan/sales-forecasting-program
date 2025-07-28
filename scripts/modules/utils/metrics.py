#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Utility functions and metrics module.
Contains helper functions, metrics calculations, and terminal capture functionality.
"""

import sys
import numpy as np
from datetime import datetime
from typing import List

# ============================================================================
# Metrics Calculations
# ============================================================================

def calculate_mape(y_true, y_pred):
    """Calculate Mean Absolute Percentage Error (MAPE)"""
    # Filter zero values
    mask = y_true != 0
    if mask.sum() == 0:
        return np.inf
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def calculate_wape(y_true, y_pred):
    """Calculate Weighted Absolute Percentage Error (WAPE)"""
    if y_true.sum() == 0:
        return np.inf
    return np.sum(np.abs(y_true - y_pred)) / np.sum(y_true) * 100

# ============================================================================
# Terminal Output Capture
# ============================================================================

# Global terminal outputs list
terminal_outputs: List[str] = []

class TerminalCapture:
    """Capture terminal outputs for PDF reporting"""
    def __init__(self):
        self.contents = []
        
    def write(self, text):
        self.contents.append(text)
        sys.__stdout__.write(text)  # Still show in terminal
        
    def flush(self):
        sys.__stdout__.flush()
        
    def get_contents(self):
        return ''.join(self.contents)

# Initialize terminal capture
terminal_capture = TerminalCapture()

def initialize_terminal_capture():
    """Initialize terminal output capture"""
    global terminal_capture
    terminal_capture = TerminalCapture()
    sys.stdout = terminal_capture

def log_output(message: str):
    """Log message to both terminal and capture"""
    print(message)
    terminal_outputs.append(f"{datetime.now().strftime('%H:%M:%S')} - {message}")

def get_terminal_contents():
    """Get captured terminal contents"""
    return terminal_capture.get_contents()

def get_terminal_outputs():
    """Get all logged outputs"""
    return terminal_outputs.copy()

def clear_terminal_outputs():
    """Clear terminal outputs list"""
    global terminal_outputs
    terminal_outputs.clear()
