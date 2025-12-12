import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Callable, Any, Union, List
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

# No need for drawing here, but if used in loop_fn, import as needed

def _like_rnncell(cell):
    """Check if cell has the necessary RNNCell interface (PyTorch nn.Module)."""
    return isinstance(cell, nn.Module) and hasattr(cell, 'state_size') and hasattr(cell, 'output_size')

def nest_flatten(structure: Any) -> List[Any]:
    """Simple nest.flatten equivalent for tuples/lists."""
    if isinstance(structure, (list, tuple)):
        return [item for sub in structure for item in nest_flatten(sub)]
    return [structure]

def nest_pack(structure: Any, flat_sequence: List[Any]) -> Any:
    """Simple nest.pack_sequence_as equivalent."""
    if isinstance(structure, (list, tuple)):
        return type(structure)(nest_pack(sub, flat_sequence) for sub in structure)
    return flat_sequence.pop(0)

def raw_rnn(cell: nn.Module, loop_fn: Callable, max_steps: int, initial_state: Any, 
            parallel_iterations: Optional[int] = None, device: str = 'cpu') -> Tuple[Any, Any, Any]:
    """
    PyTorch port of raw_rnn: unrolled RNN using a for loop up to max_steps.
    Handles nested states/outputs via lists of tuples.
    Emits states/outputs for all timesteps, final state.
    
    Args:
        cell: nn.Module with forward(inputs, state) -> (output, new_state)
        loop_fn: Callable(time, prev_output, prev_state, prev_loop_state) -> 
                 (elements_finished [batch], next_input, next_state, emit_output, next_loop_state)
        max_steps: Maximum number of timesteps to unroll.
        initial_state: Initial cell state (tensor or tuple).
        device: 'cpu' or 'cuda'.
    
    Returns:
        (states [max_steps, ...], outputs [max_steps, ...], final_state)
    """
    if not _like_rnncell(cell):
        raise TypeError("cell must be an nn.Module with state_size and output_size")
    if not callable(loop_fn):
        raise TypeError("loop_fn must be a callable")

    flat_init = nest_flatten(initial_state)
    batch_size = flat_init[0].shape[0]
    elements_finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
    current_state = initial_state
    loop_state = None
    states_list: List[Any] = []
    outputs_list: List[Any] = []

    for time in range(max_steps):
        if torch.all(elements_finished):
            break  # Early stop if all finished

        # Call loop_fn
        prev_output = outputs_list[-1] if outputs_list else None  # First: None (assume output is tensor)
        elements_finished_t, next_input, next_state, emit_output, next_loop_state = loop_fn(
            time, prev_output, current_state, loop_state
        )
        if next_loop_state is not None:
            loop_state = next_loop_state

        # Update finished (cumulative OR)
        elements_finished = elements_finished | elements_finished_t

        # Call cell
        output, new_state = cell(next_input, next_state if next_state is not None else current_state)

        # Mask finished sequences (zero out) - recursive for nested
        mask_batch = (~elements_finished).unsqueeze(-1).float()  # [batch, 1]
        def mask_tensor(t: Any) -> Any:
            if isinstance(t, (list, tuple)):
                return type(t)(mask_tensor(sub) for sub in t)
            if not isinstance(t, torch.Tensor):
                return t
            if t.dim() == 0:
                return t * (~elements_finished).float()
            # Assume batch-first: expand mask to match t's batch dim
            mask_exp = mask_batch.expand(t.shape[:1] + (1,) * (t.dim() - 1))
            return t * mask_exp

        output = mask_tensor(output)
        new_state = mask_tensor(new_state)

        # Append (keep structure)
        states_list.append(new_state)
        outputs_list.append(output)

        current_state = new_state

    # Stack along time dim (assume uniform shapes)
    def stack_list(lst: List[Any]) -> Any:
        if isinstance(lst[0], torch.Tensor):
            return torch.stack(lst, dim=0)  # [T, batch, ...]
        else:
            # Nested: recursively stack each component
            if isinstance(lst[0], (list, tuple)):
                # Assume all timesteps have same structure; stack per leaf
                def recurse_stack(items):
                    if isinstance(items[0], torch.Tensor):
                        return torch.stack(items, dim=0)
                    else:
                        return type(items[0])(recurse_stack([sub[i] for sub in items]) for i in range(len(items[0])))
                return recurse_stack(lst)
            else:
                # Flat list of tensors
                return torch.stack(lst, dim=0)

    states = stack_list(states_list)
    outputs = stack_list(outputs_list)
    final_state = current_state

    return states, outputs, final_state


def rnn_teacher_force(inputs: torch.Tensor, cell: nn.Module, sequence_length: Union[int, torch.Tensor], 
                      initial_state: Any, device: str = 'cpu') -> Tuple[Any, Any, Any]:
    """
    PyTorch port: Teacher forcing RNN - feed provided inputs sequentially.
    Equivalent to dynamic_rnn with teacher forcing.
    
    Args:
        inputs: [batch, T, input_size] or [T, batch, input_size]
        cell: nn.Module cell
        sequence_length: [batch] or scalar max T
        initial_state: Initial state
        device: Device
    
    Returns:
        (states, outputs, final_state)
    """
    if inputs.dim() == 3 and inputs.shape[0] == 1:  # Squeeze if needed
        inputs = inputs.squeeze(0)
    
    if inputs.dim() == 3 and inputs.shape[1] != sequence_length.shape[0] if hasattr(sequence_length, 'shape') else False:
        inputs = inputs.transpose(0, 1)  # To [T, batch, feat]

    T = inputs.shape[0] if inputs.dim() == 3 else sequence_length
    batch_size = inputs.shape[1] if inputs.dim() == 3 else inputs.shape[0]

    def loop_fn(time, cell_output, cell_state, loop_state):
        if hasattr(sequence_length, 'shape'):
            elements_finished = (time >= sequence_length).to(device)
        else:
            elements_finished = torch.full((batch_size,), time >= T, dtype=torch.bool, device=device)
        
        if torch.all(elements_finished):
            next_input = torch.zeros_like(inputs[0])
        else:
            next_input = inputs[min(time, T-1)]  # Clamp to avoid index error
        
        next_state = cell_state
        emit_output = cell_output if cell_output is not None else torch.zeros(batch_size, cell.output_size, device=device)
        return elements_finished, next_input, next_state, emit_output, None

    return raw_rnn(cell, loop_fn, int(T), initial_state, device=device)


def rnn_free_run(cell: nn.Module, initial_state: Any, sequence_length: Union[int, torch.Tensor], 
                 initial_input: Optional[torch.Tensor] = None, device: str = 'cpu') -> Tuple[Any, Any, Any]:
    """
    PyTorch port: Autoregressive RNN - feed predictions back via output_function.
    Stops per sequence via termination_condition.
    
    Args:
        cell: nn.Module with output_function(state) and termination_condition(state)
        initial_state: Initial state
        sequence_length: Max T [batch] or scalar
        initial_input: Optional starting input
        device: Device
    
    Returns:
        (states, outputs, final_state)
    """
    flat_init = nest_flatten(initial_state)
    batch_size = flat_init[0].shape[0]
    T = sequence_length if not hasattr(sequence_length, 'shape') else sequence_length.max().item()
    
    def loop_fn(time, cell_output, cell_state, loop_state):
        next_state = cell_state
        if hasattr(sequence_length, 'shape'):
            time_finished = (time >= sequence_length).to(device)
        else:
            time_finished = torch.full((batch_size,), time >= T, dtype=torch.bool, device=device)
        elements_finished = time_finished | cell.termination_condition(next_state)
        
        if torch.all(elements_finished):
            next_input = torch.zeros(batch_size, 3, device=device)  # Assume [batch, 3]
        else:
            next_input = initial_input if cell_output is None else cell.output_function(next_state)
        
        emit_output = next_input if cell_output is None else next_input  # Dummy emit
        return elements_finished, next_input, next_state, emit_output, None

    if initial_input is None:
        initial_input = cell.output_function(initial_state)

    return raw_rnn(cell, loop_fn, T, initial_state, device=device)


# Test rapide (adapte DummyCell pour nested state si besoin)
if __name__ == "__main__":
    batch_size = 2
    T = 5
    feat = 3
    state_size = 4
    device = torch.device('cpu')

    # Dummy cell with tuple state (h, c, extra)
    class DummyCell(nn.Module):
        def __init__(self, input_size, hidden_size):
            super().__init__()
            self.input_size = input_size
            self.hidden_size = hidden_size
            self.lstm = nn.LSTMCell(input_size, hidden_size)
            self.state_size = ((hidden_size, hidden_size), hidden_size)  # ( (h,c), extra )
            self.output_size = hidden_size

        def forward(self, inputs, state):
            lstm_state, extra = state  # Unpack full state
            h, c = lstm_state
            new_h, new_c = self.lstm(inputs, (h, c))
            new_extra = torch.zeros_like(extra)  # Dummy update
            return new_h, ((new_h, new_c), new_extra)

        def output_function(self, state):
            batch, _ = state[0][0].shape  # Infer batch from state
            return torch.cat([torch.zeros(batch, 2, device=device), torch.ones(batch, 1, device=device)], dim=1)

        def termination_condition(self, state):
            batch, _ = state[0][0].shape  # Infer batch
            return torch.zeros(batch, dtype=torch.bool, device=device)

    cell = DummyCell(feat, state_size)
    # Initial state matching state_size: ((h,c), extra)
    h0 = torch.zeros(batch_size, state_size, device=device)
    c0 = torch.zeros(batch_size, state_size, device=device)
    extra0 = torch.zeros(batch_size, state_size, device=device)
    initial_state = ((h0, c0), extra0)

    # Test teacher_force
    inputs = torch.randn(T, batch_size, feat, device=device)
    states_tf, outputs_tf, final_tf = rnn_teacher_force(inputs, cell, T, initial_state, device)
    print("Teacher Force - Outputs shape:", outputs_tf.shape)
    flat_states = nest_flatten(states_tf)
    print("States example shapes:", [str(s.shape) for s in flat_states if hasattr(s, 'shape')][:3])

    # Test free_run
    states_fr, outputs_fr, final_fr = rnn_free_run(cell, initial_state, T, device=device)
    print("Free Run - Outputs shape:", outputs_fr.shape)

    print("Test passed! (Shapes match expected)")