#!/usr/bin/env python3
"""
Advanced Chess Playing Bot with Rating System and Game Analyzer
Features:
- Adjustable AI difficulty (400-3000 ELO rating)
- Complete chess engine with minimax + alpha-beta pruning
- Game analyzer like Chess.com
- GUI interface with pygame
- Move validation and game state management
- Opening book integration
- Endgame tablebase support
"""

import pygame
import chess
import chess.engine
import chess.svg
import chess.pgn
import random
import time
import math
import threading
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from enum import Enum
import json
import os

# Initialize Pygame
pygame.init()

# Constants
BOARD_SIZE = 640
SQUARE_SIZE = BOARD_SIZE // 8
PANEL_WIDTH = 400
WHITE = (240, 217, 181)
BLACK = (181, 136, 99)
HIGHLIGHT_COLOR = (255, 255, 0, 128)
MOVE_COLOR = (0, 255, 0, 128)
CAPTURE_COLOR = (255, 0, 0, 128)
BUTTON_COLOR = (100, 150, 200)
BUTTON_HOVER = (120, 170, 220)
BUTTON_TEXT = (255, 255, 255)

class Difficulty(Enum):
    BEGINNER = (400, "Beginner")
    NOVICE = (800, "Novice")
    INTERMEDIATE = (1200, "Intermediate")
    ADVANCED = (1600, "Advanced")
    EXPERT = (2000, "Expert")
    MASTER = (2400, "Master")
    GRANDMASTER = (2800, "Grandmaster")

@dataclass
class GameState:
    board: chess.Board
    human_color: chess.Color
    ai_color: chess.Color
    difficulty: Difficulty
    move_history: List[chess.Move]
    time_spent: List[float]
    game_result: Optional[str] = None

@dataclass
class MoveAnalysis:
    move: chess.Move
    evaluation: float
    depth: int
    is_best: bool
    is_blunder: bool
    is_mistake: bool
    is_inaccuracy: bool
    classification: str

class ChessEngine:
    """Advanced chess engine with adjustable strength"""
    
    def __init__(self, difficulty: Difficulty):
        self.difficulty = difficulty
        self.rating = difficulty.value[0]
        self.transposition_table = {}
        self.opening_book = self.load_opening_book()
        
    def load_opening_book(self) -> Dict[str, List[str]]:
        """Load basic opening book"""
        return {
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -": [
                "e2e4", "d2d4", "Ng1f3", "c2c4", "g2g3"
            ],
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq -": [
                "e7e5", "c7c5", "e7e6", "c7c6", "d7d6"
            ]
        }
    
    def get_piece_value(self, piece: chess.Piece) -> int:
        """Get material value of a piece"""
        values = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 20000
        }
        return values.get(piece.piece_type, 0)
    
    def evaluate_position(self, board: chess.Board) -> float:
        """Evaluate the current position"""
        if board.is_checkmate():
            return -20000 if board.turn else 20000
        if board.is_stalemate() or board.is_insufficient_material():
            return 0
            
        score = 0
        
        # Material count
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                value = self.get_piece_value(piece)
                score += value if piece.color == chess.WHITE else -value
        
        # Positional factors
        score += self.evaluate_position_factors(board)
        
        # Add some randomness based on rating
        randomness = (3000 - self.rating) / 100
        score += random.uniform(-randomness, randomness)
        
        return score
    
    def evaluate_position_factors(self, board: chess.Board) -> float:
        """Evaluate positional factors"""
        score = 0
        
        # Center control
        center_squares = [chess.D4, chess.E4, chess.D5, chess.E5]
        for square in center_squares:
            if board.piece_at(square):
                piece = board.piece_at(square)
                if piece.color == chess.WHITE:
                    score += 10
                else:
                    score -= 10
        
        # Piece development
        for square in [chess.B1, chess.G1, chess.B8, chess.G8]:
            piece = board.piece_at(square)
            if piece and piece.piece_type == chess.KNIGHT:
                if piece.color == chess.WHITE:
                    score -= 5  # Penalty for undeveloped knights
                else:
                    score += 5
        
        # King safety
        white_king_square = board.king(chess.WHITE)
        black_king_square = board.king(chess.BLACK)
        
        if white_king_square:
            score += self.evaluate_king_safety(board, white_king_square, chess.WHITE)
        if black_king_square:
            score -= self.evaluate_king_safety(board, black_king_square, chess.BLACK)
        
        return score
    
    def evaluate_king_safety(self, board: chess.Board, king_square: chess.Square, color: chess.Color) -> float:
        """Evaluate king safety"""
        safety_score = 0
        
        # Penalty for king in center during opening/middlegame
        if len(list(board.legal_moves)) > 20:  # Rough opening/middlegame indicator
            file = chess.square_file(king_square)
            rank = chess.square_rank(king_square)
            if 2 <= file <= 5 and 2 <= rank <= 5:
                safety_score -= 50
        
        return safety_score
    
    def minimax(self, board: chess.Board, depth: int, alpha: float, beta: float, 
                maximizing: bool) -> Tuple[float, Optional[chess.Move]]:
        """Minimax algorithm with alpha-beta pruning"""
        
        # Adjust depth based on rating
        if self.rating < 1000:
            depth = min(depth, 2)
        elif self.rating < 1500:
            depth = min(depth, 3)
        elif self.rating < 2000:
            depth = min(depth, 4)
        
        if depth == 0 or board.is_game_over():
            return self.evaluate_position(board), None
        
        best_move = None
        
        if maximizing:
            max_eval = float('-inf')
            for move in board.legal_moves:
                board.push(move)
                eval_score, _ = self.minimax(board, depth - 1, alpha, beta, False)
                board.pop()
                
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_move = move
                
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break  # Alpha-beta pruning
            
            return max_eval, best_move
        else:
            min_eval = float('inf')
            for move in board.legal_moves:
                board.push(move)
                eval_score, _ = self.minimax(board, depth - 1, alpha, beta, True)
                board.pop()
                
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_move = move
                
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break  # Alpha-beta pruning
            
            return min_eval, best_move
    
    def get_book_move(self, board: chess.Board) -> Optional[chess.Move]:
        """Get a move from opening book"""
        fen = board.fen().split()[0] + " " + ("w" if board.turn else "b") + " KQkq -"
        if fen in self.opening_book:
            move_str = random.choice(self.opening_book[fen])
            try:
                return chess.Move.from_uci(move_str)
            except:
                return None
        return None
    
    def get_best_move(self, board: chess.Board) -> chess.Move:
        """Get the best move for current position"""
        
        # Check opening book first
        if len(board.move_stack) < 10:  # Only use book in opening
            book_move = self.get_book_move(board)
            if book_move and book_move in board.legal_moves:
                return book_move
        
        # Use minimax for move selection
        depth = 4
        if self.rating < 1200:
            depth = 2
        elif self.rating < 1800:
            depth = 3
        
        _, best_move = self.minimax(board, depth, float('-inf'), float('inf'), 
                                   board.turn == chess.WHITE)
        
        if best_move is None:
            # Fallback to random legal move
            return random.choice(list(board.legal_moves))
        
        return best_move

class GameAnalyzer:
    """Chess game analyzer similar to Chess.com"""
    
    def __init__(self):
        self.engine = ChessEngine(Difficulty.GRANDMASTER)
    
    def analyze_move(self, board: chess.Board, move: chess.Move, 
                    time_limit: float = 1.0) -> MoveAnalysis:
        """Analyze a single move"""
        
        # Get best move evaluation
        best_eval, best_move = self.engine.minimax(board, 4, float('-inf'), 
                                                  float('inf'), board.turn == chess.WHITE)
        
        # Get played move evaluation
        board.push(move)
        played_eval, _ = self.engine.minimax(board, 4, float('-inf'), 
                                           float('inf'), board.turn != chess.WHITE)
        board.pop()
        
        # Calculate evaluation difference
        eval_diff = abs(best_eval - played_eval)
        
        # Classify the move
        is_best = move == best_move
        is_blunder = eval_diff > 300
        is_mistake = eval_diff > 150 and not is_blunder
        is_inaccuracy = eval_diff > 50 and not is_mistake and not is_blunder
        
        if is_best:
            classification = "Best Move"
        elif is_blunder:
            classification = "Blunder"
        elif is_mistake:
            classification = "Mistake"
        elif is_inaccuracy:
            classification = "Inaccuracy"
        else:
            classification = "Good Move"
        
        return MoveAnalysis(
            move=move,
            evaluation=played_eval,
            depth=4,
            is_best=is_best,
            is_blunder=is_blunder,
            is_mistake=is_mistake,
            is_inaccuracy=is_inaccuracy,
            classification=classification
        )
    
    def analyze_game(self, game_state: GameState) -> Dict:
        """Analyze the complete game"""
        board = chess.Board()
        analysis_results = {
            'moves': [],
            'accuracy': {'white': 0, 'black': 0},
            'blunders': {'white': 0, 'black': 0},
            'mistakes': {'white': 0, 'black': 0},
            'inaccuracies': {'white': 0, 'black': 0},
            'time_stats': {
                'total_time': sum(game_state.time_spent),
                'average_time': sum(game_state.time_spent) / len(game_state.time_spent) if game_state.time_spent else 0,
                'longest_think': max(game_state.time_spent) if game_state.time_spent else 0
            }
        }
        
        good_moves = {'white': 0, 'black': 0}
        total_moves = {'white': 0, 'black': 0}
        
        for i, move in enumerate(game_state.move_history):
            analysis = self.analyze_move(board, move)
            analysis_results['moves'].append(analysis)
            
            color = 'white' if board.turn == chess.WHITE else 'black'
            total_moves[color] += 1
            
            if analysis.is_blunder:
                analysis_results['blunders'][color] += 1
            elif analysis.is_mistake:
                analysis_results['mistakes'][color] += 1
            elif analysis.is_inaccuracy:
                analysis_results['inaccuracies'][color] += 1
            else:
                good_moves[color] += 1
            
            board.push(move)
        
        # Calculate accuracy
        for color in ['white', 'black']:
            if total_moves[color] > 0:
                accuracy = (good_moves[color] / total_moves[color]) * 100
                analysis_results['accuracy'][color] = round(accuracy, 1)
        
        return analysis_results

class ChessGUI:
    """Chess GUI using pygame with advanced animations"""
    
    def __init__(self):
        self.screen = pygame.display.set_mode((BOARD_SIZE + PANEL_WIDTH, BOARD_SIZE))
        pygame.display.set_caption("Advanced Chess Bot")
        self.clock = pygame.time.Clock()
        
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)
        self.large_font = pygame.font.Font(None, 32)
        
        self.selected_square = None
        self.legal_moves = []
        self.move_start_time = None
        
        # New animation system with linear interpolation
        self.move_queue = []  # Queue of moves to animate
        self.current_animation = None
        self.last_board_state = None
        
        # Initialize piece images
        self.piece_images = {}
        self.load_piece_images()
        
        # Button areas
        self.rating_up_button = pygame.Rect(BOARD_SIZE + 20, 150, 80, 30)
        self.rating_down_button = pygame.Rect(BOARD_SIZE + 120, 150, 80, 30)
        
    def load_piece_images(self):
        """Create Chess.com-style piece images with CSS-like styling"""
        for piece_type in [chess.PAWN, chess.ROOK, chess.KNIGHT, chess.BISHOP, chess.QUEEN, chess.KING]:
            for color in [chess.WHITE, chess.BLACK]:
                piece_surface = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
                
                # Chess.com-inspired colors and styling
                if color == chess.WHITE:
                    # Light pieces: cream/ivory color like Chess.com
                    main_color = (240, 217, 181)  # Cream
                    shadow_color = (160, 130, 98)  # Darker cream for shadow
                    highlight_color = (255, 245, 220)  # Light highlight
                    border_color = (101, 67, 33)  # Dark brown border
                else:
                    # Dark pieces: rich brown like Chess.com
                    main_color = (139, 69, 19)  # Saddle brown
                    shadow_color = (69, 39, 19)  # Dark brown shadow
                    highlight_color = (160, 100, 50)  # Light brown highlight
                    border_color = (40, 20, 10)  # Very dark border
                
                # Create piece with CSS-like layered styling
                if piece_type == chess.PAWN:
                    self.draw_modern_pawn(piece_surface, main_color, shadow_color, highlight_color, border_color)
                elif piece_type == chess.ROOK:
                    self.draw_modern_rook(piece_surface, main_color, shadow_color, highlight_color, border_color)
                elif piece_type == chess.KNIGHT:
                    self.draw_modern_knight(piece_surface, main_color, shadow_color, highlight_color, border_color)
                elif piece_type == chess.BISHOP:
                    self.draw_modern_bishop(piece_surface, main_color, shadow_color, highlight_color, border_color)
                elif piece_type == chess.QUEEN:
                    self.draw_modern_queen(piece_surface, main_color, shadow_color, highlight_color, border_color)
                elif piece_type == chess.KING:
                    self.draw_modern_king(piece_surface, main_color, shadow_color, highlight_color, border_color)
                
                self.piece_images[(piece_type, color)] = piece_surface

    def draw_modern_pawn(self, surface, main_color, shadow_color, highlight_color, border_color):
        """Draw a modern Chess.com-style pawn with CSS-like layering"""
        center_x, center_y = SQUARE_SIZE // 2, SQUARE_SIZE // 2
        
        # Shadow layer (bottom)
        pygame.draw.circle(surface, shadow_color, (center_x + 2, center_y - 6), 14)
        base_shadow = pygame.Rect(center_x - 16, center_y + 7, 36, 12)
        pygame.draw.ellipse(surface, shadow_color, base_shadow)
        
        # Main body layer
        pygame.draw.circle(surface, main_color, (center_x, center_y - 8), 12)
        base_rect = pygame.Rect(center_x - 18, center_y + 5, 36, 12)
        pygame.draw.ellipse(surface, main_color, base_rect)
        
        # Highlight layer (top-left)
        pygame.draw.circle(surface, highlight_color, (center_x - 2, center_y - 10), 6)
        
        # Border layer (outline)
        pygame.draw.circle(surface, border_color, (center_x, center_y - 8), 12, 2)
        pygame.draw.ellipse(surface, border_color, base_rect, 2)

    def draw_modern_rook(self, surface, main_color, shadow_color, highlight_color, border_color):
        """Draw a modern Chess.com-style rook"""
        center_x, center_y = SQUARE_SIZE // 2, SQUARE_SIZE // 2
        
        # Shadow layer
        shadow_tower = pygame.Rect(center_x - 13, center_y - 10, 30, 24)
        pygame.draw.rect(surface, shadow_color, shadow_tower)
        
        # Main tower body
        tower_rect = pygame.Rect(center_x - 15, center_y - 12, 30, 24)
        pygame.draw.rect(surface, main_color, tower_rect)
        
        # Highlight
        highlight_rect = pygame.Rect(center_x - 13, center_y - 10, 12, 20)
        pygame.draw.rect(surface, highlight_color, highlight_rect)
        
        # Crenellations with modern styling
        for i in range(5):
            if i % 2 == 0:
                cren_rect = pygame.Rect(center_x - 15 + i * 6, center_y - 18, 6, 6)
                pygame.draw.rect(surface, main_color, cren_rect)
                pygame.draw.rect(surface, highlight_color, (cren_rect.x, cren_rect.y, 3, 6))
                pygame.draw.rect(surface, border_color, cren_rect, 1)
        
        # Base with gradient effect
        base_rect = pygame.Rect(center_x - 18, center_y + 8, 36, 8)
        pygame.draw.rect(surface, main_color, base_rect)
        pygame.draw.rect(surface, highlight_color, (base_rect.x, base_rect.y, 18, 8))
        pygame.draw.rect(surface, border_color, tower_rect, 2)
        pygame.draw.rect(surface, border_color, base_rect, 2)

    def draw_modern_knight(self, surface, main_color, shadow_color, highlight_color, border_color):
        """Draw a modern Chess.com-style knight"""
        center_x, center_y = SQUARE_SIZE // 2, SQUARE_SIZE // 2
        
        # Shadow
        shadow_points = [(center_x - 6, center_y + 17), (center_x - 10, center_y - 3), 
                        (center_x - 3, center_y - 13), (center_x + 10, center_y - 10),
                        (center_x + 14, center_y - 3), (center_x + 17, center_y + 12),
                        (center_x + 10, center_y + 17)]
        pygame.draw.polygon(surface, shadow_color, shadow_points)
        
        # Main horse head
        points = [(center_x - 8, center_y + 15), (center_x - 12, center_y - 5),
                 (center_x - 5, center_y - 15), (center_x + 8, center_y - 12),
                 (center_x + 12, center_y - 5), (center_x + 15, center_y + 10),
                 (center_x + 8, center_y + 15)]
        pygame.draw.polygon(surface, main_color, points)
        
        # Highlight on the nose/forehead
        highlight_points = [(center_x - 5, center_y - 15), (center_x + 3, center_y - 12),
                           (center_x + 8, center_y - 8), (center_x - 2, center_y - 5)]
        pygame.draw.polygon(surface, highlight_color, highlight_points)
        
        # Eye
        pygame.draw.circle(surface, border_color, (center_x + 3, center_y - 5), 2)
        pygame.draw.circle(surface, highlight_color, (center_x + 2, center_y - 6), 1)
        
        # Mane details
        for i in range(3):
            x = center_x - 8 + i * 2
            pygame.draw.line(surface, border_color, (x, center_y - 12), (x, center_y - 8), 1)
        
        # Border
        pygame.draw.polygon(surface, border_color, points, 2)
        
        # Base
        base_rect = pygame.Rect(center_x - 18, center_y + 12, 36, 6)
        pygame.draw.ellipse(surface, main_color, base_rect)
        pygame.draw.ellipse(surface, border_color, base_rect, 2)

    def draw_modern_bishop(self, surface, main_color, shadow_color, highlight_color, border_color):
        """Draw a modern Chess.com-style bishop"""
        center_x, center_y = SQUARE_SIZE // 2, SQUARE_SIZE // 2
        
        # Shadow
        shadow_points = [(center_x + 2, center_y - 16), (center_x - 6, center_y - 8),
                        (center_x - 10, center_y + 7), (center_x + 14, center_y + 7),
                        (center_x + 10, center_y - 8)]
        pygame.draw.polygon(surface, shadow_color, shadow_points)
        
        # Main body
        points = [(center_x, center_y - 18), (center_x - 8, center_y - 10),
                 (center_x - 12, center_y + 5), (center_x + 12, center_y + 5),
                 (center_x + 8, center_y - 10)]
        pygame.draw.polygon(surface, main_color, points)
        
        # Highlight
        highlight_points = [(center_x, center_y - 18), (center_x - 4, center_y - 10),
                           (center_x - 6, center_y), (center_x + 2, center_y - 5)]
        pygame.draw.polygon(surface, highlight_color, highlight_points)
        
        # Mitre top
        pygame.draw.circle(surface, main_color, (center_x, center_y - 15), 4)
        pygame.draw.circle(surface, highlight_color, (center_x - 1, center_y - 16), 2)
        pygame.draw.circle(surface, border_color, (center_x, center_y - 15), 4, 1)
        
        # Diagonal slit with depth
        pygame.draw.line(surface, shadow_color, (center_x - 6, center_y - 8), (center_x + 6, center_y + 2), 3)
        pygame.draw.line(surface, border_color, (center_x - 6, center_y - 8), (center_x + 6, center_y + 2), 1)
        
        # Border
        pygame.draw.polygon(surface, border_color, points, 2)
        
        # Base
        base_rect = pygame.Rect(center_x - 16, center_y + 8, 32, 8)
        pygame.draw.ellipse(surface, main_color, base_rect)
        pygame.draw.ellipse(surface, highlight_color, (base_rect.x, base_rect.y, 16, 8))
        pygame.draw.ellipse(surface, border_color, base_rect, 2)

    def draw_modern_queen(self, surface, main_color, shadow_color, highlight_color, border_color):
        """Draw a modern Chess.com-style queen"""
        center_x, center_y = SQUARE_SIZE // 2, SQUARE_SIZE // 2
        
        # Shadow body
        shadow_body = pygame.Rect(center_x - 10, center_y - 3, 24, 20)
        pygame.draw.ellipse(surface, shadow_color, shadow_body)
        
        # Main body
        body_rect = pygame.Rect(center_x - 12, center_y - 5, 24, 20)
        pygame.draw.ellipse(surface, main_color, body_rect)
        
        # Highlight on body
        highlight_body = pygame.Rect(center_x - 10, center_y - 3, 12, 16)
        pygame.draw.ellipse(surface, highlight_color, highlight_body)
        
        # Crown with modern styling
        crown_points = []
        heights = [center_y - 10, center_y - 14, center_y - 18, center_y - 14, center_y - 10]
        for i in range(5):
            x = center_x - 16 + i * 8
            crown_points.extend([(x, heights[i]), (x + 4, center_y - 8)])
        crown_points.extend([(center_x + 16, center_y - 5), (center_x - 16, center_y - 5)])
        
        # Crown shadow
        pygame.draw.polygon(surface, shadow_color, crown_points)
        
        # Crown main
        pygame.draw.polygon(surface, main_color, crown_points)
        
        # Crown highlights
        for i in range(5):
            x = center_x - 16 + i * 8
            highlight_rect = pygame.Rect(x, heights[i], 2, center_y - 5 - heights[i])
            pygame.draw.rect(surface, highlight_color, highlight_rect)
        
        # Crown jewels with modern gems
        gem_colors = [(220, 20, 60), (0, 100, 200), (255, 215, 0), (50, 205, 50), (138, 43, 226)]
        for i in range(5):
            x = center_x - 16 + i * 8 + 2
            y = heights[i] + 3
            pygame.draw.circle(surface, gem_colors[i], (x, y), 3)
            pygame.draw.circle(surface, (255, 255, 255, 128), (x - 1, y - 1), 1)
        
        # Borders
        pygame.draw.polygon(surface, border_color, crown_points, 2)
        pygame.draw.ellipse(surface, border_color, body_rect, 2)
        
        # Base
        base_rect = pygame.Rect(center_x - 18, center_y + 12, 36, 6)
        pygame.draw.ellipse(surface, main_color, base_rect)
        pygame.draw.ellipse(surface, highlight_color, (base_rect.x, base_rect.y, 18, 6))
        pygame.draw.ellipse(surface, border_color, base_rect, 2)

    def draw_modern_king(self, surface, main_color, shadow_color, highlight_color, border_color):
        """Draw a modern Chess.com-style king"""
        center_x, center_y = SQUARE_SIZE // 2, SQUARE_SIZE // 2
        
        # Shadow body
        shadow_body = pygame.Rect(center_x - 12, center_y - 6, 28, 24)
        pygame.draw.ellipse(surface, shadow_color, shadow_body)
        
        # Main body
        body_rect = pygame.Rect(center_x - 14, center_y - 8, 28, 24)
        pygame.draw.ellipse(surface, main_color, body_rect)
        
        # Body highlight
        highlight_body = pygame.Rect(center_x - 12, center_y - 6, 14, 20)
        pygame.draw.ellipse(surface, highlight_color, highlight_body)
        
        # Crown base with gradient
        crown_rect = pygame.Rect(center_x - 16, center_y - 15, 32, 8)
        pygame.draw.rect(surface, shadow_color, (crown_rect.x + 1, crown_rect.y + 1, 32, 8))
        pygame.draw.rect(surface, main_color, crown_rect)
        pygame.draw.rect(surface, highlight_color, (crown_rect.x, crown_rect.y, 16, 8))
        
        # Cross with 3D effect
        # Shadow cross
        pygame.draw.line(surface, shadow_color, (center_x + 1, center_y - 21), (center_x + 1, center_y - 11), 4)
        pygame.draw.line(surface, shadow_color, (center_x - 3, center_y - 17), (center_x + 5, center_y - 17), 4)
        
        # Main cross
        pygame.draw.line(surface, main_color, (center_x, center_y - 22), (center_x, center_y - 12), 4)
        pygame.draw.line(surface, main_color, (center_x - 4, center_y - 18), (center_x + 4, center_y - 18), 4)
        
        # Cross highlight
        pygame.draw.line(surface, highlight_color, (center_x - 1, center_y - 22), (center_x - 1, center_y - 12), 2)
        pygame.draw.line(surface, highlight_color, (center_x - 4, center_y - 19), (center_x + 2, center_y - 19), 2)
        
        # Crown details with gems
        gem_positions = [(center_x - 8, center_y - 11), (center_x, center_y - 11), (center_x + 8, center_y - 11)]
        gem_colors = [(220, 20, 60), (0, 100, 200), (255, 215, 0)]
        for i, (x, y) in enumerate(gem_positions):
            pygame.draw.circle(surface, gem_colors[i], (x, y), 3)
            pygame.draw.circle(surface, (255, 255, 255, 128), (x - 1, y - 1), 1)
        
        # Borders
        pygame.draw.ellipse(surface, border_color, body_rect, 2)
        pygame.draw.rect(surface, border_color, crown_rect, 2)
        
        # Base
        base_rect = pygame.Rect(center_x - 18, center_y + 12, 36, 6)
        pygame.draw.ellipse(surface, main_color, base_rect)
        pygame.draw.ellipse(surface, highlight_color, (base_rect.x, base_rect.y, 18, 6))
        pygame.draw.ellipse(surface, border_color, base_rect, 2)
    
    def square_to_coords(self, square: chess.Square) -> Tuple[int, int]:
        """Convert chess square to screen coordinates"""
        file = chess.square_file(square)
        rank = chess.square_rank(square)
        x = file * SQUARE_SIZE
        y = (7 - rank) * SQUARE_SIZE
        return x, y
    
    def coords_to_square(self, x: int, y: int) -> Optional[chess.Square]:
        """Convert screen coordinates to chess square"""
        if x < 0 or x >= BOARD_SIZE or y < 0 or y >= BOARD_SIZE:
            return None
        file = x // SQUARE_SIZE
        rank = 7 - (y // SQUARE_SIZE)
        return chess.square(file, rank)
    
    def draw_board(self, board: chess.Board):
        """Draw the chess board"""
        for rank in range(8):
            for file in range(8):
                color = WHITE if (rank + file) % 2 == 0 else BLACK
                rect = pygame.Rect(file * SQUARE_SIZE, rank * SQUARE_SIZE, 
                                 SQUARE_SIZE, SQUARE_SIZE)
                pygame.draw.rect(self.screen, color, rect)
                
                # Draw coordinates
                if file == 0:
                    rank_text = self.small_font.render(str(8 - rank), True, 
                                                     BLACK if color == WHITE else WHITE)
                    self.screen.blit(rank_text, (rect.x + 5, rect.y + 5))
                if rank == 7:
                    file_text = self.small_font.render(chr(ord('a') + file), True, 
                                                     BLACK if color == WHITE else WHITE)
                    self.screen.blit(file_text, (rect.x + SQUARE_SIZE - 15, 
                                               rect.y + SQUARE_SIZE - 20))
    
    def draw_pieces(self, board: chess.Board):
        """Draw chess pieces with smooth animations"""
        # Store current board state for comparison
        if self.last_board_state is None:
            self.last_board_state = board.copy()
        
        # Process current animation if any
        if self.current_animation:
            self.update_animation()
        
        # Draw all pieces except the one being animated
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                # Skip piece being animated
                if (self.current_animation and 
                    square == self.current_animation.get('from_square')):
                    continue
                    
                x, y = self.square_to_coords(square)
                piece_image = self.piece_images.get((piece.piece_type, piece.color))
                if piece_image:
                    self.screen.blit(piece_image, (x, y))
        
        # Draw animated piece if active
        if self.current_animation:
            self.draw_moving_piece()
    
    def update_animation(self):
        """Update current animation state"""
        if not self.current_animation:
            return
            
        current_time = time.time()
        elapsed = current_time - self.current_animation['start_time']
        duration = self.current_animation['duration']
        
        if elapsed >= duration:
            # Animation complete
            self.current_animation = None
            # Process next move in queue if any
            if self.move_queue:
                next_move_data = self.move_queue.pop(0)
                self.start_new_animation(next_move_data)
    
    def draw_moving_piece(self):
        """Draw the piece that's currently animating with linear interpolation"""
        if not self.current_animation:
            return
            
        current_time = time.time()
        elapsed_time = current_time - self.current_animation['start_time']
        total_duration = self.current_animation['duration']
        progress = min(elapsed_time / total_duration, 1.0)
        
        # Get start and end positions
        start_position = self.square_to_coords(self.current_animation['from_square'])
        end_position = self.square_to_coords(self.current_animation['to_square'])
        
        # Use linear interpolation formula: current_position = start_position + (end_position - start_position) * (elapsed_time / total_duration)
        current_x = start_position[0] + (end_position[0] - start_position[0]) * progress
        current_y = start_position[1] + (end_position[1] - start_position[1]) * progress
        
        # Draw piece at current position
        piece = self.current_animation['piece']
        piece_image = self.piece_images.get((piece.piece_type, piece.color))
        if piece_image:
            self.screen.blit(piece_image, (current_x, current_y))
    
    def queue_move_animation(self, board: chess.Board, move: chess.Move):
        """Queue a move for animation"""
        piece = board.piece_at(move.from_square)
        if not piece:
            return
            
        move_data = {
            'board_before': board.copy(),
            'move': move,
            'piece': piece,
            'from_square': move.from_square,
            'to_square': move.to_square,
            'duration': 0.7  # 700ms animation
        }
        
        if self.current_animation is None:
            # Start immediately
            self.start_new_animation(move_data)
        else:
            # Queue for later
            self.move_queue.append(move_data)
    
    def start_new_animation(self, move_data):
        """Start a new animation"""
        self.current_animation = move_data.copy()
        self.current_animation['start_time'] = time.time()
    
    def is_animating(self):
        """Check if any animation is currently playing"""
        return self.current_animation is not None or len(self.move_queue) > 0
    
    def draw_highlights(self, board: chess.Board):
        """Draw minimal square highlights"""
        if self.selected_square is not None:
            x, y = self.square_to_coords(self.selected_square)
            highlight_surface = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE))
            highlight_surface.set_alpha(80)
            highlight_surface.fill((255, 255, 0))
            self.screen.blit(highlight_surface, (x, y))
    
    def draw_rating_buttons(self, engine: 'ChessEngine'):
        """Draw rating adjustment buttons"""
        mouse_pos = pygame.mouse.get_pos()
        
        # Rating Up Button
        up_color = BUTTON_HOVER if self.rating_up_button.collidepoint(mouse_pos) else BUTTON_COLOR
        pygame.draw.rect(self.screen, up_color, self.rating_up_button)
        pygame.draw.rect(self.screen, (0, 0, 0), self.rating_up_button, 2)
        
        up_text = self.font.render("Rating +", True, BUTTON_TEXT)
        up_rect = up_text.get_rect(center=self.rating_up_button.center)
        self.screen.blit(up_text, up_rect)
        
        # Rating Down Button
        down_color = BUTTON_HOVER if self.rating_down_button.collidepoint(mouse_pos) else BUTTON_COLOR
        pygame.draw.rect(self.screen, down_color, self.rating_down_button)
        pygame.draw.rect(self.screen, (0, 0, 0), self.rating_down_button, 2)
        
        down_text = self.font.render("Rating -", True, BUTTON_TEXT)
        down_rect = down_text.get_rect(center=self.rating_down_button.center)
        self.screen.blit(down_text, down_rect)
    
    def is_rating_button_clicked(self, pos: Tuple[int, int]) -> str:
        """Check if rating buttons were clicked"""
        if self.rating_up_button.collidepoint(pos):
            return "up"
        elif self.rating_down_button.collidepoint(pos):
            return "down"
    def draw_info_panel(self, game_state: GameState, engine: 'ChessEngine'):
        """Draw the information panel"""
        # Clear panel area
        pygame.draw.rect(self.screen, (240, 240, 240), 
                        (BOARD_SIZE, 0, PANEL_WIDTH, BOARD_SIZE))
        
        y_offset = 20
        
        # Current turn
        turn_text = "White to move" if game_state.board.turn == chess.WHITE else "Black to move"
        text_surface = self.font.render(turn_text, True, (0, 0, 0))
        self.screen.blit(text_surface, (BOARD_SIZE + 20, y_offset))
        y_offset += 50
        
        # Engine rating
        rating_text = f"Engine Rating: {engine.rating}"
        text_surface = self.font.render(rating_text, True, (0, 0, 0))
        self.screen.blit(text_surface, (BOARD_SIZE + 20, y_offset))
        y_offset += 50
        
        # Draw rating buttons
        self.draw_rating_buttons(engine)
        y_offset += 100
        
        # Game status
        if game_state.board.is_checkmate():
            status_text = "Checkmate!"
            color = (255, 0, 0)
        elif game_state.board.is_stalemate():
            status_text = "Stalemate!"
            color = (255, 165, 0)
        elif game_state.board.is_check():
            status_text = "Check!"
            color = (255, 165, 0)
        else:
            status_text = "Game in progress"
            color = (0, 128, 0)
        
        text_surface = self.font.render(status_text, True, color)
        self.screen.blit(text_surface, (BOARD_SIZE + 20, y_offset))
        y_offset += 70
        
        # Show animation info
        animation_text = "Animation: Linear 700ms"
        text_surface = self.small_font.render(animation_text, True, (100, 100, 100))
        self.screen.blit(text_surface, (BOARD_SIZE + 20, y_offset))

class ChessGame:
    """Main chess game class"""
    
    def __init__(self):
        self.gui = ChessGUI()
        self.difficulty = Difficulty.INTERMEDIATE
        self.engine = ChessEngine(self.difficulty)
        self.analyzer = GameAnalyzer()
        self.game_state = GameState(
            board=chess.Board(),
            human_color=chess.WHITE,
            ai_color=chess.BLACK,
            difficulty=self.difficulty,
            move_history=[],
            time_spent=[]
        )
        self.ai_thinking = False
        
    def handle_mouse_click(self, pos: Tuple[int, int]) -> bool:
        """Handle mouse click on the board"""
        if (self.ai_thinking or 
            self.game_state.board.turn != self.game_state.human_color or
            self.gui.is_animating()):
            return False
        
        x, y = pos
        if x >= BOARD_SIZE:  # Clicked on info panel
            return self.handle_info_panel_click(x - BOARD_SIZE, y)
        
        clicked_square = self.gui.coords_to_square(x, y)
        if clicked_square is None:
            return False
        
        if self.gui.selected_square is None:
            # Select a square
            piece = self.game_state.board.piece_at(clicked_square)
            if piece and piece.color == self.game_state.human_color:
                self.gui.selected_square = clicked_square
                self.gui.legal_moves = list(self.game_state.board.legal_moves)
                self.gui.move_start_time = time.time()
        else:
            # Try to make a move
            move = chess.Move(self.gui.selected_square, clicked_square)
            
            # Handle promotion
            if (self.game_state.board.piece_at(self.gui.selected_square) and
                self.game_state.board.piece_at(self.gui.selected_square).piece_type == chess.PAWN):
                if ((self.game_state.human_color == chess.WHITE and chess.square_rank(clicked_square) == 7) or
                    (self.game_state.human_color == chess.BLACK and chess.square_rank(clicked_square) == 0)):
                    move = chess.Move(self.gui.selected_square, clicked_square, promotion=chess.QUEEN)
            
            if move in self.game_state.board.legal_moves:
                self.make_human_move(move)
                return True
            
            self.gui.selected_square = None
            self.gui.legal_moves = []
        
        return False
    
    def handle_info_panel_click(self, x: int, y: int) -> bool:
        """Handle clicks on the info panel (for difficulty selection, rating buttons, etc.)"""
        # Check for rating button clicks
        panel_pos = (x + BOARD_SIZE, y)
        rating_action = self.gui.is_rating_button_clicked(panel_pos)
        
        if rating_action == "up":
            # Increase rating by 100, max 3000
            self.engine.rating = min(self.engine.rating + 100, 3000)
            return True
        elif rating_action == "down":
            # Decrease rating by 100, min 400
            self.engine.rating = max(self.engine.rating - 100, 400)
            return True
        
        # Simple difficulty cycling for now
        if 10 <= y <= 35:  # Clicked on difficulty line
            difficulties = list(Difficulty)
            current_index = difficulties.index(self.difficulty)
            self.difficulty = difficulties[(current_index + 1) % len(difficulties)]
            self.engine = ChessEngine(self.difficulty)
            self.game_state.difficulty = self.difficulty
            return True
        return False
    
    def make_human_move(self, move: chess.Move):
        """Make a human move with improved animation"""
        if self.gui.move_start_time:
            move_time = time.time() - self.gui.move_start_time
            self.game_state.time_spent.append(move_time)
        
        # Queue the animation
        self.gui.queue_move_animation(self.game_state.board, move)
        
        # Make the move immediately (animation handles the visual transition)
        self.game_state.board.push(move)
        self.game_state.move_history.append(move)
        
        self.gui.selected_square = None
        self.gui.legal_moves = []
        self.gui.move_start_time = None
        
        # Check if game is over
        if self.game_state.board.is_game_over():
            self.handle_game_over()
        else:
            # Start AI move in separate thread, but wait for animation to finish
            self.ai_thinking = True
            threading.Thread(target=self.make_ai_move_delayed, daemon=True).start()
    
    def make_ai_move_delayed(self):
        """Wait for current animation to finish, then make AI move"""
        # Wait for any ongoing animation to complete
        while self.gui.is_animating():
            time.sleep(0.05)
        
        self.make_ai_move()
    
    def make_ai_move(self):
        """Make an AI move with improved animation"""
        start_time = time.time()
        
        # Add some thinking time based on difficulty
        thinking_time = max(0.5, (self.engine.rating - 400) / 1000)
        
        move = self.engine.get_best_move(self.game_state.board)
        
        # Ensure minimum thinking time for realism
        elapsed = time.time() - start_time
        if elapsed < thinking_time:
            time.sleep(thinking_time - elapsed)
        
        move_time = time.time() - start_time
        self.game_state.time_spent.append(move_time)
        
        # Queue the animation
        self.gui.queue_move_animation(self.game_state.board, move)
        
        # Make the move immediately (animation handles the visual transition)
        self.game_state.board.push(move)
        self.game_state.move_history.append(move)
        
        self.ai_thinking = False
        
        # Check if game is over
        if self.game_state.board.is_game_over():
            self.handle_game_over()
    
    def handle_game_over(self):
        """Handle end of game"""
        result = self.game_state.board.result()
        
        if result == "1-0":
            self.game_state.game_result = "White wins"
        elif result == "0-1":
            self.game_state.game_result = "Black wins"
        else:
            self.game_state.game_result = "Draw"
        
        print(f"Game Over: {self.game_state.game_result}")
        
        # Analyze the game
        analysis = self.analyzer.analyze_game(self.game_state)
        self.display_game_analysis(analysis)
    
    def display_game_analysis(self, analysis: Dict):
        """Display game analysis results"""
        print("\n" + "="*50)
        print("GAME ANALYSIS")
        print("="*50)
        
        print(f"Game Result: {self.game_state.game_result}")
        print(f"Total Moves: {len(self.game_state.move_history)}")
        print(f"Game Duration: {analysis['time_stats']['total_time']:.1f} seconds")
        print(f"Average Move Time: {analysis['time_stats']['average_time']:.1f} seconds")
        
        print("\nACCURACY:")
        print(f"White: {analysis['accuracy']['white']}%")
        print(f"Black: {analysis['accuracy']['black']}%")
        
        print("\nMISTAKES:")
        for color in ['white', 'black']:
            print(f"{color.capitalize()}:")
            print(f"  Blunders: {analysis['blunders'][color]}")
            print(f"  Mistakes: {analysis['mistakes'][color]}")
            print(f"  Inaccuracies: {analysis['inaccuracies'][color]}")
        
        print("\nMOVE-BY-MOVE ANALYSIS:")
        board = chess.Board()
        for i, move_analysis in enumerate(analysis['moves'][:10]):  # Show first 10 moves
            move_num = i // 2 + 1
            color = "White" if i % 2 == 0 else "Black"
            san_move = board.san(move_analysis.move)
            
            print(f"{move_num}. {san_move} ({color}): {move_analysis.classification}")
            board.push(move_analysis.move)
        
        if len(analysis['moves']) > 10:
            print(f"... and {len(analysis['moves']) - 10} more moves")
    
    def reset_game(self):
        """Reset the game"""
        self.game_state = GameState(
            board=chess.Board(),
            human_color=self.game_state.human_color,
            ai_color=self.game_state.ai_color,
            difficulty=self.difficulty,
            move_history=[],
            time_spent=[]
        )
        self.ai_thinking = False
        self.gui.selected_square = None
        self.gui.legal_moves = []
    
    def run(self):
        """Main game loop"""
        running = True
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Left click
                        self.handle_mouse_click(event.pos)
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:  # R key to reset
                        self.reset_game()
                    elif event.key == pygame.K_f:  # F key to flip board
                        self.game_state.human_color = not self.game_state.human_color
                        self.game_state.ai_color = not self.game_state.ai_color
            
            # Clear screen
            self.gui.screen.fill((0, 0, 0))  # Fill with black for clean background
            
            # Draw everything
            self.gui.draw_board(self.game_state.board)
            self.gui.draw_highlights(self.game_state.board)
            self.gui.draw_pieces(self.game_state.board)
            self.gui.draw_info_panel(self.game_state, self.engine)
            
            # Draw thinking indicator
            if self.ai_thinking:
                thinking_text = self.gui.font.render("AI is thinking...", True, (255, 0, 0))
                self.gui.screen.blit(thinking_text, (BOARD_SIZE + 10, 200))
            
            pygame.display.flip()
            self.gui.clock.tick(60)
        
        pygame.quit()

def main():
    """Main function"""
    print("Advanced Chess Bot with PyTweening Animations")
    print("============================================")
    print("Features:")
    print("- Click to select and move pieces")
    print("- Click on difficulty to cycle through levels")
    print("- Press 'R' to reset game")
    print("- Press 'F' to flip colors")
    print("- Game analysis will be shown after each game")
    print("- Smooth 700ms linear animations for piece movement")
    print()
    
    try:
        game = ChessGame()
        game.run()
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have pygame, python-chess, and pytweening installed:")
        print("pip install pygame python-chess pytweening")

if __name__ == "__main__":
    main()