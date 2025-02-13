"""
A library for exploring generative grammars and their connection to compression.
Contains classes and functions for creating and analyzing grammars, with examples
from the Fibonacci sequence, language patterns, and Pong game mechanics.

The code is structured as both a library and a runnable example file.
Uncomment the examples at the bottom to run them.
"""

import dataclasses
from typing import List, Dict, Tuple, Optional
import math


@dataclasses.dataclass
class Rule:
    """A production rule in a context-free grammar."""
    left: str
    right: List[str]
    
    def __str__(self):
        return f"{self.left} -> {' '.join(self.right)}"


class Grammar:
    """A context-free grammar with production rules."""
    
    def __init__(self, rules: List[Rule], start_symbol: str = 'S'):
        self.rules = rules
        self.start_symbol = start_symbol
        
    def description_length(self) -> int:
        """Calculate the description length of the grammar in bits."""
        total_length = 0
        # For each rule, we need to encode the left side and right side symbols
        for rule in self.rules:
            # Assume each symbol takes log2(num_unique_symbols) bits to encode
            unique_symbols = set([rule.left] + rule.right)
            bits_per_symbol = math.log2(len(unique_symbols))
            total_length += (1 + len(rule.right)) * bits_per_symbol
        return math.ceil(total_length)
    
    def print_rules(self):
        """Print all rules in the grammar."""
        print("\nGrammar Rules:")
        print("--------------")
        for rule in self.rules:
            print(str(rule))


def generate_fibonacci(n: int) -> List[int]:
    """Generate first n Fibonacci numbers."""
    sequence = [1, 1]
    for _ in range(n-2):
        sequence.append(sequence[-1] + sequence[-2])
    return sequence


class PongGrammar:
    """A grammar for generating Pong game states and visuals."""
    
    def __init__(self):
        self.rules = [
            Rule('GAME', ['INIT', 'LOOP']),
            Rule('INIT', ['Ball(center)', 'Paddle1(mid)', 'Paddle2(mid)', 'Score(0,0)']),
            Rule('LOOP', ['UPDATE', 'COLLISIONS', 'SCORE', 'LOOP']),
            Rule('UPDATE', ['Ball(pos += vel)', 'Paddle1(pos += input1)', 'Paddle2(pos += input2)']),
            Rule('COLLISIONS', ['WALL_CHECK', 'PADDLE_CHECK']),
            Rule('WALL_CHECK', ['if ball.y > height: ball.vel.y *= -1']),
            Rule('PADDLE_CHECK', ['if ball.collides(paddle): ball.vel.x *= -1']),
            Rule('SCORE', ['if ball.x < 0: p2_score++ else if ball.x > width: p1_score++']),
        ]
        self.grammar = Grammar(self.rules)


# Example 1: Fibonacci Sequence Compression
"""
# Uncomment to run the Fibonacci sequence example

def demonstrate_fibonacci_compression():
    # Generate first 100 Fibonacci numbers
    fib_sequence = generate_fibonacci(100)
    print("First 100 Fibonacci numbers:")
    print(", ".join(str(x) for x in fib_sequence[:10]) + "...")
    print(f"100th Fibonacci number: {fib_sequence[-1]:,}")
    
    # Calculate direct storage needs
    direct_bits = sum(math.ceil(math.log2(x)) for x in fib_sequence)
    print(f"\nDirect storage needs {direct_bits:,} bits")
    
    # Naive Grammar
    naive_rules = [Rule('S', ['F1'])] + [
        Rule(f'F{i+1}', [str(num)])
        for i, num in enumerate(fib_sequence)
    ]
    naive_grammar = Grammar(naive_rules)
    naive_bits = naive_grammar.description_length()
    
    # Pattern Recognition Grammar
    pattern_rules = [
        Rule('S', ['F', 'N1']),
        Rule('N1', ['1']),
    ] + [
        Rule(f'N{fib}', [str(fib)])
        for fib in sorted(set(fib_sequence[:20]))
    ] + [
        Rule('F', ['N1']),
        Rule('F', ['N1', '+', 'N1']),
        Rule('F', ['N2', '+', 'N3']),
        Rule('F', ['N3', '+', 'N5']),
        Rule('F', ['N5', '+', 'N8']),
    ]
    pattern_grammar = Grammar(pattern_rules)
    pattern_bits = pattern_grammar.description_length()
    
    # Abstract Grammar
    abstract_rules = [
        Rule('S', ['F', '1']),
        Rule('F', ['1']),
        Rule('F', ['F', '+', 'F_prev'])
    ]
    abstract_grammar = Grammar(abstract_rules)
    abstract_bits = abstract_grammar.description_length()
    
    # Print results
    print("\nCompression Comparison:")
    print("=" * 50)
    print(f"Direct storage:     {direct_bits:6,} bits (baseline)")
    print(f"Naive grammar:      {naive_bits:6,} bits ({direct_bits/naive_bits:6.2f}x better)")
    print(f"Pattern grammar:    {pattern_bits:6,} bits ({direct_bits/pattern_bits:6.2f}x better)")
    print(f"Abstract grammar:   {abstract_bits:6,} bits ({direct_bits/abstract_bits:6.2f}x better)")
"""

# Example 2: Language Pattern Compression
"""
# Uncomment to run the language pattern example

def demonstrate_language_compression():
    sentences = [
        "I like cats and dogs",
        "I like books and music",
        "I like coffee and tea",
        "I like movies and games"
    ]
    direct_bits = sum(len(s) * 8 for s in sentences)  # 8 bits per char
    print(f"Direct storage needs {direct_bits} bits")
    
    # Naive Grammar
    naive_rules = [
        Rule('S', ['I', 'like', 'cats', 'and', 'dogs']),
        Rule('S', ['I', 'like', 'books', 'and', 'music']),
        Rule('S', ['I', 'like', 'coffee', 'and', 'tea']),
        Rule('S', ['I', 'like', 'movies', 'and', 'games'])
    ]
    naive_grammar = Grammar(naive_rules)
    naive_bits = naive_grammar.description_length()
    
    # Pattern Recognition Grammar
    pattern_rules = [
        Rule('S', ['I', 'like', 'THING', 'and', 'THING']),
        Rule('THING', ['cats']), Rule('THING', ['dogs']),
        Rule('THING', ['books']), Rule('THING', ['music']),
        Rule('THING', ['coffee']), Rule('THING', ['tea']),
        Rule('THING', ['movies']), Rule('THING', ['games'])
    ]
    pattern_grammar = Grammar(pattern_rules)
    pattern_bits = pattern_grammar.description_length()
    
    # Abstract Grammar
    abstract_rules = [
        Rule('S', ['I', 'like', 'N1', 'and', 'N2']),
        Rule('N1', ['cats', 'books', 'coffee', 'movies']),
        Rule('N2', ['dogs', 'music', 'tea', 'games'])
    ]
    abstract_grammar = Grammar(abstract_rules)
    abstract_bits = abstract_grammar.description_length()
    
    # Print results
    print("\nLanguage Pattern Compression:")
    print("=" * 50)
    print(f"Direct storage:     {direct_bits:6,} bits (baseline)")
    print(f"Naive grammar:      {naive_bits:6,} bits ({direct_bits/naive_bits:6.2f}x better)")
    print(f"Pattern grammar:    {pattern_bits:6,} bits ({direct_bits/pattern_bits:6.2f}x better)")
    print(f"Abstract grammar:   {abstract_bits:6,} bits ({direct_bits/abstract_bits:6.2f}x better)")
"""

# Example 3: Pong Game State Compression
"""
# Uncomment to run the Pong game state example

def demonstrate_pong_compression():
    # Game parameters
    frames = 300 * 60  # 5 minutes at 60 FPS
    frame_height = 210
    frame_width = 160
    channels = 3  # RGB
    
    # Calculate sizes
    pong = PongGrammar()
    pos_frame_size = (4 + 4 + 8)  # 4 bytes each for paddles, 8 for ball position
    pos_raw_size = pos_frame_size * frames
    grammar_size = pong.grammar.description_length()
    input_size = 2 * frames  # 1 bit per paddle per frame
    state_size = 16  # 4 bytes each for ball x,y and two paddle positions
    
    # Calculate full visual data size
    pixel_frame_size = frame_height * frame_width * channels
    pixel_raw_size = pixel_frame_size * frames
    
    # Print results
    print("\nPong Game State Compression:")
    print("=" * 50)
    print("Conservative Estimate (Position Data Only)")
    print(f"Raw position data:   {pos_raw_size:,} bytes ({pos_raw_size/1024:.1f} KB)")
    print(f"Grammar + state:     {grammar_size + input_size + state_size:,} bytes")
    print(f"Compression ratio:   {pos_raw_size/(grammar_size + input_size + state_size):.2f}x")
    
    print("\nFull Visual Data Estimate")
    print(f"Raw video:          {pixel_raw_size:,} bytes ({pixel_raw_size/1024/1024:.1f} MB)")
    print(f"Grammar + state:     {grammar_size + input_size + state_size:,} bytes")
    print(f"Compression ratio:   {pixel_raw_size/(grammar_size + input_size + state_size):,.2f}x")
"""

if __name__ == "__main__":
    # Uncomment the examples you want to run
    # demonstrate_fibonacci_compression()
    # demonstrate_language_compression()
    # demonstrate_pong_compression()
    pass 