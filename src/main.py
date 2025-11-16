from .utils import chess_manager, GameContext
from numpy import binary_repr
import chess
import random
import time
import torch
from torch import nn
from huggingface_hub import hf_hub_download
from transformers import AutoModel
import os

# Write code here that runs once
#

device = ("cuda" if torch.cuda.is_available() else "cpu")

def to_binary(n):
    s = binary_repr(n)
    l = len(s)
    return s.rjust(64, '0')


#takes an int
def to_binary_vector(n):
    if n is None:
        n=0
    return torch.tensor([float(int(x)) for x in to_binary(n)])

def halfkp_tensor(board: chess.Board) -> torch.Tensor:

    print("entering halfkp_tensor...")
    NUM_PIECES = 12       # 6 piece types × 2 colors
    NUM_SQ = 64
    FEATURE_SIZE = NUM_PIECES * NUM_SQ * NUM_SQ  # 12 × 64 × 64 = 49152

    print("generating empty features vector...")
    features = torch.zeros(FEATURE_SIZE, dtype=torch.float32)

    print("getting turn and king position...")
    stm = board.turn
    king_sq = board.king(stm)

    print("checking if king_sq is none")
    if king_sq is None:
        raise ValueError("Position has no king for the side to move.")

    print("iterating over pieces")
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is None:
            continue

        
        # Do NOT encode the king of the side to move (already used as context)
        if piece.piece_type == chess.KING and piece.color == stm:
            continue

        # Map piece to 0–11
        piece_index = (piece.piece_type - 1) + 6 * piece.color

        # HalfKP feature index:
        # (piece, king_square, piece_square)
        idx = (
            piece_index * NUM_SQ * NUM_SQ
            + king_sq * NUM_SQ
            + sq
        )

        features[idx] = 1.0

    return features

#takes a board
def to_input_vector(b):
    return torch.cat([halfkp_tensor(b),
                      to_binary_vector(b.castling_rights),
                      to_binary_vector(b.ep_square),
                      torch.tensor([float(b.turn)])],
                        dim = 0
                        )

b = chess.Board()
input_length = len(to_input_vector(b))

class chessNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.bot = nn.Sequential(nn.Linear(input_length, 512), nn.ReLU(inplace = True),
                                nn.Linear(512, 32),nn.ReLU(inplace = True),
                                nn.Linear(32,32),nn.ReLU(inplace = True),
                                nn.Linear(32, 1)
                                )
    def forward(self, x):
        return self.bot(x)

magnus = chessNN()

# Download a specific file (e.g., a GGUF file)
file_path = hf_hub_download(repo_id="n-nazz/chesshacks", filename="magnus_weights", local_dir="/home/nazz/3b/chesshacks/my-chesshacks-bot/magnus_weights_download")
magnus.load_state_dict(torch.load(file_path, map_location = torch.device(device)))

def FindVal(myboard,searchDepth,fanOut):

    print("0")
    if (searchDepth == 0):
        print("\\frac{1}{2}")
        positionEncoding  = to_input_vector(myboard)

        print("\\frac{3}{4}")
        return magnus(positionEncoding)

    print("1")
    legal_moves = list(myboard.generate_legal_moves())

    if not legal_moves:
        ctx.logProbabilities({})
        if (myboard.turn):
            return -3300
        else:
            return 3300
    print("2")
    #rating each possible move
    PredMoveValues = []
    for move in legal_moves:
        myboard.push(move)

        positionEncoding  = to_input_vector(myboard)
        print("3")
        PredMoveValues.append(magnus(positionEncoding))
        print("4")
        myboard.pop()

    PredMoveValues_Tens = torch.Tensor(PredMoveValues)
    PredMoveValues_Tens.requires_grad=False
    print("requires_grad is fine")
    WhiteMoveProbs_tens = torch.nn.functional.softmax(PredMoveValues_Tens, dim=0)
    print("dim=0 also fine")
    WhiteMoveProbs = WhiteMoveProbs_tens.tolist()
    print("5")
    # Normalize so probabilities sum to 1


    if (myboard.turn):
        move_probs = [
             WhiteMoveProb
            for move, WhiteMoveProb in zip(legal_moves, WhiteMoveProbs)
        ]
    else:
        move_probs = [
            1 - WhiteMoveProb 
            for move, WhiteMoveProb in zip(legal_moves, WhiteMoveProbs)
        ]
    chosenMoves = random.choices(legal_moves, weights=move_probs, k=fanOut)

    nextPositionValues = []
    print("6")
    for move in chosenMoves:
        print("pushing in for loop")
        myboard.push(move)

        nextPositionValues.append(FindVal(myboard,searchDepth-1,fanOut))

        myboard.pop()
    print("7 (past for loop)")
    
    nextPositionValues = torch.Tensor(nextPositionValues)
    
    if (myboard.turn):
        return nextPositionValues[torch.argmax(nextPositionValues)]
    else:
        return nextPositionValues[torch.argmax(nextPositionValues)]


@chess_manager.entrypoint
def test_func(ctx: GameContext):
    # This gets called every time the model needs to make a move
    # Return a python-chess Move object that is a legal move for the current position

    b = chess.Board()
    bullshit = to_input_vector(b)
    print("Cooking move...")
    print(ctx.board.move_stack)

    legal_moves = list(ctx.board.generate_legal_moves())
    if not legal_moves:
        ctx.logProbabilities({})
        raise ValueError("No legal moves available (i probably lost didn't i)")
    print("generated legal move list!")
    
    searchDepth = 1
    fanOut = 5

    PredMoveValues = []
    myboard = chess.Board(ctx.board.fen())
    for move in legal_moves:
        print(f"pushing move {move} to board...")
        myboard.push(move)

        print("computing position encoding...")
        positionEncoding  = to_input_vector(myboard)

        print("about to call FindVal function")
        PredMoveValues.append(FindVal(myboard,searchDepth,fanOut))
        myboard.pop()

    print("calculated PredMoveValues successfully")

    total_value = sum(PredMoveValues)
    # Normalize so probabilities sum to 1

    print("about to normalize")

    PredMoveValues_Tens = torch.Tensor(PredMoveValues)
    PredMoveValues_Tens.requires_grad=False
    print("requires_grad is fine")
    WhiteMoveProbs_tens = torch.nn.functional.softmax(PredMoveValues_Tens, dim=0)
    print("dim=0 also fine")
    WhiteMoveProbs = WhiteMoveProbs_tens.tolist()
    print("5")
    # Normalize so probabilities sum to 1


    if (myboard.turn):
        move_probs = [
             WhiteMoveProb
            for move, WhiteMoveProb in zip(legal_moves, WhiteMoveProbs)
        ]
    else:
        move_probs = [
            1 - WhiteMoveProb 
            for move, WhiteMoveProb in zip(legal_moves, WhiteMoveProbs)
        ]
    print("if statement is ok")
    print(f"legal moves len is {len(legal_moves)} and move probs len is {len(move_probs)}.")
    print(move_probs)
    print(f"minimum of move_probs is {min(move_probs)}")
    if(len(legal_moves)==1):
        move_probs[0]=1
    chosenMove = random.choices(legal_moves, weights=move_probs, k=1)[0]
    print("returning!")
    return chosenMove


@chess_manager.reset
def reset_func(ctx: GameContext):
    # This gets called when a new game begins
    # Should do things like clear caches, reset model state, etc.
    pass

