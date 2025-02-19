---
title : The Pāṇinian Approach to Compression
date : 11/02/2025
---
> "The simplest of several competing explanations is likely to be the correct one" - Occam's Razor

## Intelligence as compression

(Skip to [code](/notes/panini/#pāṇinis-razor) walkthrough.)

Philosopher John Searle in his famous [Chinese Room argument](https://plato.stanford.edu/archIves/spr2010/entries/chinese-room/) argues against the possibility of a computer ever being able to think. From outside the closed room it may look like it's conversing in Chinese, but could the machine inside with a memorized table of what Chinese word comes after what could be considered to have understood Chinese? The argument intends to show that while suitably programmed computers may appear to converse, they are not capable of developing any understanding of meaning or semantics, even in principle. 

Searle's argument is incorrect. We will soon see how.

Today's most prominent AIs, large language models (LLMs) like DeepSeek, GPT, or Claude are more sophisticated versions of such shallow 'generative' memory models. They use thousands of tokens of context and the 'attention' mechanism to focus on relevant parts of the input to 'query' a table with.  They're not *just* storing a memorized table of what comes after what, but using compute to extract and store a hiearchy of reusable information in their layers. A result is that the weights of the Llama-65B model occupy around 365GB on disk, down from the 5.6TB it's trained on **(a 14x compression)**[1]. We see that generalization ability and data efficiency are equivalent: generalization comes from squeezing every bit of information out of your datapoints, 'understanding' all correlations and causations, and connecting all the dots. "Squeezing every bit of information" is meant literally: generalization is the very direct result of compression. 

One could argue against calling this process of parroting statistical regularities 'true understanding'. A 14x (or so) compression ratio would be fairly impressive if nothing better existed. However, these machine learning algorithms are many orders-of-magnitude less data efficient than human beings. Lee Sedol, a top Go player, played around 10,000 games in his lifetime, while DeepMind's bot AlphaGo bot required 30 million games to match him (powered by the energy requirements of a small city). If Sedol had played 30 million games, how skilled would he be? What would a human who has absorbed all of human knowledge look like? What sort of "information squeezing" is the human brain doing so effectively? I am convinced that the answer to these questions is the key to building general machine intelligence i.e. machines that think, learn, adapt to tasks like (or better than) humans do.

But it's hard to run this experiment because most humans have seen orders of magnitude less data than any LLM in their lifetimes, and it's hard to manually inspect human priors. Or so I thought, until I attended the lecture series[2] by Dr. Saroja Bhate at Bangalore International Centre, on Pāṇini (pronounced "pah-nee-nee"), the ancient Sanskrit grammarian. Over 2300 years ago, before the advent of computers or formal logic, Pāṇini sat down and methodically reduced all of human knowledge, then floating around in spoken Vedic Sanskrit, into a generative grammar of exactly 3,995 *sūtras*, or rewrite rules - recorded in his magnum opus, the *Aṣṭādhyāyī*. These rules have remained unchanged ever since. 

Faced with a large corpus of spoken Vedic and contemporary Sanskrit, many thousands of hours of audio signals collected without any substrate to record with or automated tooling to work with, Pāṇini, over 12 years, found abstracted atoms of meaning that when combined with a set of dynamic rules and meta-rules formed a generative grammar, a deterministic state machine that could be used to re-synthesize the original audio corpus - and be recited in just 2 hours (an astonishing compression ratio of atleast ~5000:1)[3]. 

His work is the first formal system known to man, doing to linguistic reality what Euclid would go on to do later for geometry, but it would be no overstatement to recognize it the only example I've seen of true optimal [Solomonoff induction](https://en.m.wikipedia.org/wiki/Solomonoff%27s_theory_of_inductive_inference) - finding the shortest unchanging executable archive of a dataset, as evidenced by it's durability over two millenia. 

Indeed, the very existence of Pāṇini and his work disproves Searle's argument. Yes the machine in Searle's room doesn't understand Chinese, but there is someone in the system who does - the hidden compressor that created the rule table for the machine to lookup (a grammarian like Pāṇini). The compressor applied the dual techniques of abstraction and economy ruthlessly to thousands of noisy signals of various forms and fidelity, reducing them by many orders of magnitude into succinct set of formal predicates - deriving a fixed set of rules, an explanation unchanging in time. Like breaking down a house into basic Lego-like blocks and then building a new house from it back gain, the compressor like this could then combine these discovered reusable concepts on the fly using abstracted transformation rules to generalize to any unknown. Tomorrow if the raw source signals changed, the compressor would have a method to refactor the rules entirely for this new version of reality. The compressor is where the "understanding" is, the process of compression is where the true "intelligence" in the closed system resides. 

When we talk about building intelligent machines, it is indeed building this compressor which we must talk about, NOT the fixed-in-time rules it discovers (a hand-written program, or a set of weights in a neural network). A digital superintelligence in action, would very much look like a grammarian on steroids, ruthlessly employing what we describe above as "Pāṇini's Razor". It is the efficient, automated grammarian which is intelligent, not the ever-updating grammar it generated. Of course, Pāṇini was working with Sanskrit, whose underlying structure makes it less context-sensitive than English and more amenable to such decomposition. But for the sake of comparison, his methods if automated and applied to the [Hutter compression prize](http://prize.hutter1.net/) dataset could compress 1GB of Wikipedia data to a few kilobytes (down from the current record of 110MB as of Feb 2025). 

Searle would perhaps not have made the Chinese Room argument had he heard of Pāṇini's techniques. To build thinking machines of the future, we must not repeat his mistake. 

## Pāṇini's Razor

I have no formal training in Sanskrit, and the following is merely a programmer's attempt to reverse-engineer Pāṇini's methods of compression. This is largely a guide for other programmers, so we dive into code right away. Let's first look at how generative grammars compress information. All mistakes are mine.

Imagine you're trying to send the first 100 Fibonacci numbers to a friend. The naive approach would be to simply transmit these numbers directly. Let's see just how big this sequence gets:

```python
def generate_fibonacci(n: int) -> List[int]:
    """Generate first n Fibonacci numbers."""
    sequence = [1, 1]
    for _ in range(n-2):
        sequence.append(sequence[-1] + sequence[-2])
    return sequence

fib_sequence = generate_fibonacci(100)
print("First 100 Fibonacci numbers:")
print(", ".join(str(x) for x in fib_sequence[:10]) + "...")
print(f"100th Fibonacci number: {fib_sequence[-1]:,}")
direct_bits = sum(math.ceil(math.log2(x)) for x in fib_sequence)
print(f"\nDirect storage needs {direct_bits:,} bits")
```
```
First 100 Fibonacci numbers:
1, 1, 2, 3, 5, 8, 13, 21, 34, 55...
100th Fibonacci number: 354,224,848,179,261,915,075

Direct storage needs 2,649 bits
```

That's a lot of information! The numbers get very large very quickly - the 100th Fibonacci number is a 21-digit number. Let's see how different approaches to compression handle this much larger sequence.

## The Building Blocks

We'll use two main classes to explore this idea:
- `Rule`: Represents a production rule in our grammar, like "S -> A B" meaning "S can be replaced with A followed by B"
- `Grammar`: A collection of rules that can generate patterns, with methods to calculate how many bits we need to store it

You can find the implementation details in [`grammar.py`](/code/grammar.py).

## Approach 1: Naive Grammar - Simple Memorization

Let's start with the most straightforward approach - just writing down what we see:

```python
# Create a rule for each Fibonacci number
naive_rules = [Rule('S', ['F1'])] + [
    Rule(f'F{i+1}', [str(num)])
    for i, num in enumerate(fib_sequence)
]
naive_grammar = Grammar(naive_rules)
naive_bits = naive_grammar.description_length()
print(f"Naive grammar needs {naive_bits:,} bits")
print(f"Compression ratio: {direct_bits/naive_bits:.2f}x")
```
```
Naive grammar needs 892 bits
Compression ratio: 2.97x
```

This represents the most basic level of pattern recognition:
- ✓ Noticing that numbers can be labeled (F1, F2, etc.)
- ✓ Understanding sequence order
- ✗ No understanding of relationships between numbers
- ✗ No recognition of the underlying pattern

It's like a student who memorizes "1, 1, 2, 3, 5, 8..." without understanding why these numbers appear in this order. They achieve some compression just by being organized, but they're still essentially memorizing. Suppose we have 5.6TB of text data, ethically scraped from the internet, broken into word pairs (2-grams) stored in a lookup table. When asked to complete "I was going to wear a...", it might meaninglessly output "a lot" because "a lot" appears more frequently than "a shirt" or "a skirt". A 3-gram model, using two words of context, improves accuracy but still fails in cases like "It's raining outside, wear a...". A memorized table like this would be an example of a generative models - predictingthe statistically-most-likely next word based on patterns in the data.

## Approach 2: Pattern Recognition - Seeing Relationships

Now we start to notice relationships between numbers. This requires more sophisticated observation:

```python
pattern_rules = [
    Rule('S', ['F', 'N1']),
    Rule('N1', ['1']),
] + [
    Rule(f'N{fib}', [str(fib)])
    for fib in sorted(set(fib_sequence[:20]))  # First 20 unique numbers
] + [
    Rule('F', ['N1']),
    Rule('F', ['N1', '+', 'N1']),  # 1 + 1 = 2
    Rule('F', ['N2', '+', 'N3']),  # 2 + 3 = 5
    Rule('F', ['N3', '+', 'N5']),  # 3 + 5 = 8
    Rule('F', ['N5', '+', 'N8']),  # 5 + 8 = 13
]
pattern_grammar = Grammar(pattern_rules)
pattern_bits = pattern_grammar.description_length()
print(f"Pattern grammar needs {pattern_bits:,} bits")
print(f"Compression ratio: {direct_bits/pattern_bits:.2f}x")
```
```
Pattern grammar needs 428 bits
Compression ratio: 6.19x
```

This represents an intermediate level of understanding:
- ✓ Recognition that numbers are related through addition
- ✓ Ability to see specific instances of the pattern
- ✗ Still manually writing out each addition step
- ✗ No recognition of the recursive nature
- ✗ Can't generate numbers beyond what's explicitly coded

It's like a student who realizes "Oh, I can get each number by adding specific previous numbers!" They're starting to see relationships, but they're still writing out each step manually. They might even make a table:
```
1 + 1 = 2
2 + 3 = 5
3 + 5 = 8
```
This is better than memorization, but they haven't yet had their "aha!" moment.

## Approach 3: Full Abstraction - The Cognitive Leap

Finally, we make the cognitive leap to understand the deep structure:

```python
abstract_rules = [
    Rule('S', ['F', '1']),
    Rule('F', ['1']),
    Rule('F', ['F', '+', 'F_prev'])
]
abstract_grammar = Grammar(abstract_rules)
abstract_bits = abstract_grammar.description_length()
print(f"Abstract grammar needs {abstract_bits:,} bits")
print(f"Compression ratio: {direct_bits/abstract_bits:.2f}x")
```
```
Abstract grammar needs 14 bits
Compression ratio: 189.21x
```

This represents the highest level of understanding, requiring several cognitive breakthroughs:
- ✓ Recognition that each number depends on the previous TWO numbers
- ✓ Understanding that this single relationship explains EVERY number
- ✓ Grasping that you don't need to store the numbers themselves
- ✓ Realizing the pattern is recursive and self-contained
- ✓ Understanding that this works for ANY length sequence

The cognitive steps to reach this understanding typically involve:
1. Noticing that you're always adding two numbers
2. Realizing those two numbers are always the previous ones
3. The key insight: this ONE rule explains EVERYTHING
4. Understanding that with just this rule and a starting point, you can generate the entire sequence

It's like the student who suddenly exclaims "Wait... we're ALWAYS just adding the last two numbers! That's all we need to know!" This is the moment of true understanding, where the pattern becomes crystal clear and beautifully simple.

## Generating the Sequence

The beauty of this abstract grammar is that it's not just for compression - we can use it to regenerate the original sequence or extend it to any length we want:

```python
def generate_fibonacci(n: int) -> List[int]:
    """Generate first n Fibonacci numbers using our grammar rules."""
    sequence = [1, 1]  # Initial state from our grammar's rules
    for _ in range(n-2):
        sequence.append(sequence[-1] + sequence[-2])  # F -> F + F_prev rule
    return sequence

# Test the generator
print("Regenerating our sequence:")
print(generate_fibonacci(10))  # First 10 numbers
print("\nExtending beyond what we originally stored:")
print(generate_fibonacci(15))  # First 15 numbers
```
```
Regenerating our sequence:
[1, 1, 2, 3, 5, 8, 13, 21, 34, 55]

Extending beyond what we originally stored:
[1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610]
```

This demonstrates the true power of understanding the generative process: with just 14 bits of grammar rules, we can:
1. Reproduce the original sequence exactly
2. Generate any Fibonacci number we want
3. Extend the sequence infinitely

The massive improvement in compression ratio (189.21x vs 2.97x) reflects this deep understanding. We've gone from:
- Memorizing 100 numbers (naive) →
- Understanding specific additions (pattern) →
- Grasping the universal rule (abstract)

This progression mirrors how humans learn: from memorization, to pattern recognition, to true understanding. The fact that better understanding leads to better compression isn't just a mathematical curiosity - it's a fundamental principle of how we make sense of the world.

Let's visualize these compression ratios:

```python
print("\nCompression Comparison:")
print("=" * 50)
print(f"Direct storage:     {direct_bits:6,} bits (baseline)")
print(f"Naive grammar:      {naive_bits:6,} bits ({direct_bits/naive_bits:6.2f}x better)")
print(f"Pattern grammar:    {pattern_bits:6,} bits ({direct_bits/pattern_bits:6.2f}x better)")
print(f"Abstract grammar:   {abstract_bits:6,} bits ({direct_bits/abstract_bits:6.2f}x better)")
```
```
Compression Comparison:
==================================================
Direct storage:      2,649 bits (baseline)
Naive grammar:         892 bits (  2.97x better)
Pattern grammar:       428 bits (  6.19x better)
Abstract grammar:       14 bits (189.21x better)
```

This dramatic improvement shows the true power of finding the underlying generative process. The more data we have, the more valuable it becomes to understand the true pattern rather than just storing or partially compressing the output.

## From Numbers to Language

Let's apply the same thinking to language patterns. Here's a set of similar sentences:

```python
sentences = [
    "I like cats and dogs",
    "I like books and music",
    "I like coffee and tea",
    "I like movies and games"
]
direct_bits = sum(len(s) * 8 for s in sentences)  # 8 bits per char
print(f"Direct storage needs {direct_bits} bits")
```
```
Direct storage needs 688 bits
```

That's a lot of bits! Let's try our different approaches:

### Naive Grammar (Store Each Sentence):
```python
naive_rules = [
    Rule('S', ['I', 'like', 'cats', 'and', 'dogs']),
    Rule('S', ['I', 'like', 'books', 'and', 'music']),
    Rule('S', ['I', 'like', 'coffee', 'and', 'tea']),
    Rule('S', ['I', 'like', 'movies', 'and', 'games'])
]
naive_grammar = Grammar(naive_rules)
naive_bits = naive_grammar.description_length()
print(f"Naive grammar needs {naive_bits} bits (compression ratio: {direct_bits/naive_bits:.2f}x)")
```
```
Naive grammar needs 63 bits (compression ratio: 10.92x)
```

Just by recognizing words as reusable symbols, we get almost 11x compression!

### Pattern Recognition (Find Common Structure):
```python
pattern_rules = [
    Rule('S', ['I', 'like', 'THING', 'and', 'THING']),
    Rule('THING', ['cats']), Rule('THING', ['dogs']),
    Rule('THING', ['books']), Rule('THING', ['music']),
    Rule('THING', ['coffee']), Rule('THING', ['tea']),
    Rule('THING', ['movies']), Rule('THING', ['games'])
]
pattern_grammar = Grammar(pattern_rules)
pattern_bits = pattern_grammar.description_length()
print(f"Pattern grammar needs {pattern_bits} bits (compression ratio: {direct_bits/pattern_bits:.2f}x)")
```
```
Pattern grammar needs 30 bits (compression ratio: 22.93x)
```

By recognizing that we can reuse the pattern "I like X and Y", we get even better compression!

### Full Abstraction (Separate First and Second Items):
```python
abstract_rules = [
    Rule('S', ['I', 'like', 'N1', 'and', 'N2']),
    Rule('N1', ['cats', 'books', 'coffee', 'movies']),
    Rule('N2', ['dogs', 'music', 'tea', 'games'])
]
abstract_grammar = Grammar(abstract_rules)
abstract_bits = abstract_grammar.description_length()
print(f"Abstract grammar needs {abstract_bits} bits (compression ratio: {direct_bits/abstract_bits:.2f}x)")
```
```
Abstract grammar needs 39 bits (compression ratio: 17.64x)
```

Interestingly, trying to be too clever (separating first and second items) actually hurts our compression! This is a key insight: the best compression comes from matching the true structure of the data.

### The Challenge of English Grammar

However, English presents a challenge for such clean abstractions. Unlike our previous examples, English is a context-sensitive language, which means the interpretation of a word or phrase often depends on its context. For example:

```python
context_examples = [
    "The bank is closed" ,           # Financial institution
    "The bank is muddy",            # River bank
    "I bank on your support",       # Rely/depend
    "The plane will bank left"      # Aviation term
]
```

This context sensitivity means the same word can have different meanings based on surrounding words, the same grammatical structure can have different interpretations, and valid combinations depend on semantic meaning, not just syntax.  

This limits how much we can compress English using pure grammatical rules. Our compression ratio of 22.93x was possible because we used a very restricted subset of English. In the general case, we would need better rules about how context influences meaning, better word sense disambiguation etc.

This is why formal languages (like programming languages) and some natural languages with more rigid structure achieve better compression ratios.

### A More Compressible Language

some more ancient languages like Sanskrit (and Greek) turn out to have a structure more amenable to rule-based generation. This isbecause:
1. Words are derived from root forms using explicit rules
2. Compound words follow clear compositional rules
3. Sentence structure has more rigid constraints
4. Word meanings are more systematically related to their roots

This makes a language like Sanskrit more like our Fibonacci sequence - there are clear generative rules that can produce valid constructions.

### Panini's Method in Action

Let's look at some concrete examples of how Panini's grammar generates Sanskrit words and sentences:

1. **Root-Based Word Generation**:
   ```python
   # Example: Generating forms of "bhū" (to be)
   root_rules = [
       Rule('VERB', ['ROOT', 'SUFFIX']),
       Rule('ROOT', ['bhū']),  # "to be"
       Rule('SUFFIX', ['ami']),  # 1st person present
       Rule('SUFFIX', ['asi']),  # 2nd person present
       Rule('SUFFIX', ['ati'])   # 3rd person present
   ]
   ```
   This generates:
   - bhavāmi (I am)
   - bhavasi (you are)
   - bhavati (he/she/it is)

   The transformation bhū → bhav is handled by another rule (guṇa strengthening). When the root 'bhū' combines with certain suffixes, this rule changes the 'ū' sound to 'av', demonstrating how Panini's system handles systematic sound (phonological) changes that occur when morphemes combine.

2. **Compound Word Formation**:
   ```python
   # Example: Generating compound words
   compound_rules = [
       Rule('COMPOUND', ['WORD1', 'WORD2']),
       Rule('WORD1', ['rāja']),  # king
       Rule('WORD2', ['putra'])  # son
   ]
   sandhi_rules = [
       Rule('SANDHI', ['a', 'a'], ['ā']),  # a + a → ā
   ]
   ```
   This generates:
   - rāja + putra → rājaputra (king's son)
   The rules handle both combination and sound changes (sandhi).

3. **Sentence Structure**:
   ```python
   # Example: Generating active/passive sentences
   sentence_rules = [
       Rule('S', ['NP', 'VP']),
       Rule('NP', ['devadatta']),  # Devadatta (name)
       Rule('VP', ['odanaṃ', 'pacati']),  # rice + cooks
       Rule('VP_PASSIVE', ['odanaḥ', 'pacyate'])  # rice + is cooked
   ]
   ```
   This generates:
   - devadattaḥ odanaṃ pacati (Devadatta cooks rice)
   - odanaḥ pacyate (Rice is cooked)

The power of Panini's system comes from how these rules interact:

1. **Recursive Application**:
   ```python
   # Example: Complex word formation
   derivation_rules = [
       Rule('WORD', ['ROOT', 'PRIMARY_SUFFIX', 'SECONDARY_SUFFIX']),
       Rule('ROOT', ['bhū']),
       Rule('PRIMARY_SUFFIX', ['ana']),  # action noun
       Rule('SECONDARY_SUFFIX', ['tva'])  # abstract quality
   ]
   ```
   This generates:
   - bhū → bhavana (becoming) → bhavanatva (the quality of becoming)

2. **Meta-Rules**:
   ```python
   # Example: Rule ordering
   meta_rules = [
       Rule('APPLY_ORDER', ['ROOT_RULES', 'SANDHI_RULES', 'ACCENT_RULES']),
       Rule('EXCEPTION', ['if_final_position', 'skip_sandhi'])
   ]
   ```
These meta-rules ensure correct application order and handle exceptions systematically.

We see Panini's approach compresses his original corpus by many orders of magnitude by systematically doing *more* with *less*:
1. Each rule can generate many forms (e.g., one verb root rule generates hundreds of conjugations)
2. Rules interact to produce complex forms (like compounds and derivatives)
3. Meta-rules handle exceptions without needing separate rules for each case
4. The system captures both form (phonology) and meaning (semantics)

For example, from just the root "bhū" and a set of rules, Panini's grammar can generate:
- All conjugated forms (bhavāmi, bhavasi, bhavati, etc.)
- All derived nouns (bhavana, bhāva, bhūti, etc.)
- All compounds (bhūloka - world of existence, bhūtapūrva - having been before, etc.)
- All these forms in different syntactic roles

This is analogous to our Fibonacci example, where one simple rule (Fn = Fn-1 + Fn-2) generates an infinite sequence. But Panini's grammar goes further, handling multiple interacting patterns simultaneously while maintaining semantic coherence.

This brings us to a crucial insight about building machine intelligence. The key to achieving a ~5000:1 compression ratio like Panini did, lies not in the lookup tables of Searle's Chinese Room, but in the process that created those tables - the hidden Panini-like compressor that could derive rules like the one above through the repeated application of abstraction and economy. Let's next try to understand what making an automated grammarian might look like.

## From Numbers to Panini: The First Computational Grammarian

How might Panini have done it? let's look at our own process of compressing the Fibonacci sequence:

1. Start with examples (like our Fibonacci numbers or sentences)
2. Look for patterns (like our pattern recognition phase)
3. Abstract to rules (like our final recursive sgrammars)

Panini likely followed a similar path:
1. **Data Collection**: Gathered/recorded thousands of source material (~10,000 hours of spoken Sanskrit)
2. **Pattern Recognition**: Identified recurring structures
3. **Rule Abstraction**: Derived minimal generative rules, meta-rules, and exceptions
4. **Optimization**: Compressed rules for memorization (~reduced to 2 hours of ~4000 rules that can be memorized)

In the case of the fibonacci sequence, the method achieves a compression ratio of 189.21x, in the case of Sanskrit, Panini's grammar achieves a compression ratio of ~5000:1:.

What makes Panini's work particularly relevant to our discussion is that it demonstrates the same principles we've discovered with compressing Fibonacci sequences:

1. **Cognitive Progression**: Like our Fibonacci example progressing from naive to abstract, from memorizing words to understanding derivation rules

2. **Minimal Description**: Like our abstract Fibonacci grammar using just 3 rules, his grammar captures an entire language in ~4,000 rules

3. **Generative Power**: Like our abstract grammars generating infinite sequences, his system can generate all valid Sanskrit constructions

## From Ancient Grammar to Modern Systems : The Case of Pong

While Panini's work on Sanskrit grammar might seem purely academic, this method of identifying patterns, finding minimal rules, and using them to generate valid outputs can be applied far beyond language - it provides a universal framework for understanding and compressing any complex system, allowing us to generate infinite valid states from a small set of core rules.

The real power of Panini's compression approach to generative grammars becomes clear when we apply it to modelling more complex systems with real-world state machines - like video games. Just as Panini found that all of Sanskrit could be generated from ~4,000 rules, we'll see how an entire game can be generated from just 8 rules. Let's look at how we could compress a recording of a Pong game:

```python
class PongGrammar:
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

# Let's calculate sizes for a 5-minute game at 60 FPS
frames = 300 * 60  # 5 minutes at 60 FPS

# First, let's look at a conservative estimate (just storing positions)
pos_frame_size = (4 + 4 + 8)  # 4 bytes each for paddles, 8 for ball position
pos_raw_size = pos_frame_size * frames
grammar_size = pong.grammar.description_length()  # Rules of the game
input_size = 2 * frames  # 1 bit per paddle per frame
state_size = 16  # 4 bytes each for ball x,y and two paddle positions

print("Conservative Estimate (Position Data Only)")
print("=========================================")
print(f"Raw position data:   {pos_raw_size:,} bytes ({pos_raw_size/1024:.1f} KB)")
print(f"Grammar + state:     {grammar_size + input_size + state_size:,} bytes ({(grammar_size + input_size + state_size)/1024:.1f} KB)")
print(f"Compression ratio:   {pos_raw_size/(grammar_size + input_size + state_size):.2f}x")

# Now let's look at the full visual data
frame_height = 210
frame_width = 160
channels = 3  # RGB
pixel_frame_size = frame_height * frame_width * channels  # bytes per frame
pixel_raw_size = pixel_frame_size * frames  # Full video storage

print("\nFull Visual Data Estimate (Every Pixel)")
print("=======================================")
print(f"Frame dimensions: {frame_width}x{frame_height} pixels (RGB)")
print(f"Storage requirements:")
print(f"- Raw video:        {pixel_raw_size:,} bytes ({pixel_raw_size/1024/1024:.1f} MB)")
print(f"- Grammar rules:    {grammar_size:,} bytes")
print(f"- Player inputs:    {input_size:,} bytes")
print(f"- Game state:       {state_size:,} bytes")
print(f"- Total compressed: {grammar_size + input_size + state_size:,} bytes ({(grammar_size + input_size + state_size)/1024:.1f} KB)")
print(f"Compression ratio:  {pixel_raw_size/(grammar_size + input_size + state_size):,.2f}x")

# To put this in perspective:
print("\nTo store a 5-minute Pong game:")
print("1. Conservative Approach (Just Positions)")
print(f"   - Raw data: {pos_raw_size/1024:.1f} KB")
print(f"   - Compressed: {(grammar_size + input_size + state_size)/1024:.1f} KB")
print(f"   - Ratio: {pos_raw_size/(grammar_size + input_size + state_size):.2f}x better")
print("\n2. Full Visual Approach (Every Pixel)")
print(f"   - Raw data: {pixel_raw_size/1024/1024:.1f} MB")
print(f"   - Compressed: {(grammar_size + input_size + state_size)/1024:.1f} KB")
print(f"   - Ratio: {pixel_raw_size/(grammar_size + input_size + state_size):,.2f}x better")
```
```
Conservative Estimate (Position Data Only)
=========================================
Raw position data:   1,872,000 bytes (1,828.1 KB)
Grammar + state:     38,064 bytes (37.2 KB)
Compression ratio:   49.18x

Full Visual Data Estimate (Every Pixel)
=======================================
Frame dimensions: 160x210 pixels (RGB)
Storage requirements:
- Raw video:        18,144,000,000 bytes (17,304.7 MB)
- Grammar rules:    2,048 bytes
- Player inputs:    36,000 bytes
- Game state:       16 bytes
- Total compressed: 38,064 bytes (37.2 KB)
Compression ratio:  476,671.45x

To store a 5-minute Pong game:
1. Conservative Approach (Just Positions)
   - Raw data: 1,828.1 KB
   - Compressed: 37.2 KB
   - Ratio: 49.18x better

2. Full Visual Approach (Every Pixel)"
   - Raw data: 17,304.7 MB
   - Compressed: 37.2 KB
   - Ratio: 476,671.45x better
```

This is remarkable in two ways:

1. **Conservative Estimate** (Just storing positions):
   - Raw data: ~1.8 MB
   - Compressed: ~37 KB
   - Nearly 50x compression just for the game state!

2. **Full Visual Data** (Every pixel of every frame):
   - Raw data: ~17.3 GB
   - Compressed: ~37 KB
   - Almost 500,000x compression!

Even our conservative estimate shows impressive compression because we're capturing the underlying game logic. But when we consider that this same grammar can generate the complete visual output, the compression becomes staggering. We achieve this because:

1. The rules of the game (our grammar) - about 2KB
2. The player inputs - about 35KB
3. The game state - just 16 bytes

The grammar captures both the physics engine and the rendering logic in just 8 rules! This massive compression ratio illustrates a profound point: when we truly understand a system, we don't need to store its behavior - we can store its rules and regenerate any behavior we want. This is exactly how human intelligence works: we don't memorize every position or pixel of every object we've ever seen moving - we learn the intuitive laws of physics and can use them to predict and understand any motion.

## The Connection to Intelligence

The relationship between compression and intelligence becomes clear when we look at our progression from simple sequences to complex systems. In each case, better compression came from better understanding ("modelling") of the dynamics of the system:

1. For Fibonacci, understanding the recursive relationship led to 189.21x compression
2. For the English sentences, recognizing sentence structure led to 22.93x compression (limited by English's context sensitivity)
3. For Sanskrit, Panini's grammar achieved remarkable compression of an entire language with just ~4,000 rules (~5000:1 compression ratio)
4. For Pong, applying Panini's principles to game physics led to 476,671.45x compression

This progression - from numbers to language to video games - shows how the same fundamental principles of finding minimal generative descriptions apply across domains. This is why compression can be seen as a measure of intelligence: the better we understand a system, the more efficiently we can describe it.

In machine learning terms, this is closely related to the concept of "minimum description length" (MDL) principle, which states that the best model for a dataset is the one that minimizes the size of the model (our grammar rules) and maximizes the size of the data it explains (our inputs)

This principle is formalized by Ray Solomonoff in his theory of universal inductive inference. 

> "If the universe is generated by an algorithm, then observations of that universe, encoded as a dataset, are best predicted by the smallest executable archive of that dataset"

The connection to Solomonoff induction helps explain why Panini's grammar has remained useful for over two millennia: by finding the shortest possible rule set that could generate Sanskrit, he wasn't just being clever with compression - he was discovering the true underlying structure of the language. Finding the most compressed representation of data (like Panini's grammar rules) isn't just an efficiency trick - it's mathematically optimal for prediction and understanding.

This connection between compression and understanding has profound implications:

**Learning** is essentially finding better abstract grammars for observed data

**Intelligence** can be measured by ability to find compact descriptions in the fewest observations (e.g. by looking at the least number of Fibonacci numbers)

**Understanding** means finding the true generative process that produces past and future observations

When a child learns physics, they're not memorizing the position of every object they've seen - they're learning the grammar of motion. When we understand language, we don't memorize every possible sentence - we learn the rules that generate valid ones. When Panini created his grammar, he wasn't just documenting Sanskrit - he was discovering a fundamental approach to understanding that we can apply to everything from ancient languages to modern video games.

Indeed, this is what we've demonstrated with our code examples above and what Panini achieved with Sanskrit. His work isn't just limited to modelling the spoken linguistic reality of the time - but could well be about discovering a universal principle of intelligence that will hold the key to building thinking machines of the future.
It reveals that true understand is not just in blind statistical compression, but in deducing the exact underlying processes that created that data. Whether it's a sequence of numbers, a set of sentences, or a video game, the principle remains the same: understanding the rules of generation is the key to both compression and comprehension. To understand the universe we must build a twin of it.
 
> "Riemann invented his geometries before Einstein had a use for them; the physics of our universe is not that complicated in an absolute sense.  A Bayesian superintelligence, hooked up to a webcam, would invent General Relativity as a hypothesis—perhaps not the dominant hypothesis, compared to Newtonian mechanics, but still a hypothesis under direct consideration—by the time it had seen the third frame of a falling apple.  It might guess it from the first frame, if it saw the statics of a bent blade of grass." - E. Yudkowsky

## References

[1] [Youtube video](https://www.youtube.com/watch?v=dO4TPJkeaaU), Compression for AGI, Jack Rae, Stanford MLSys, ex-OpenAI, 2023

[2] [Youtube Playlist](https://www.youtube.com/playlist?list=PLsAPTmdVuspykLNnjs1_zQKRMqRRfDr2R), Pāṇini Lecture Series, Dr. Saroja Bhate, Bangalore International Center, 2023

[3] The *Aṣṭādhyāyī* achieves a remarkable compression ratio of at least 5000:1, condensing the rules that can generate over 10,000 hours of attested Sanskrit literature (including the ~20,000 verses of the four Vedas, along with the 100,000 verses of the Mahabharata, 24,000 verses of the Ramayana, 400,000 verses of the Puranas, and hundreds of thousands of verses across texts, philosophical shastras, and classical poetry) into just 2 hours of precisely formulated rules.

[4] Pāṇini: Catching the Ocean in a Cow's Hoofprint, Vikram Chandra, 2019[blog.granthika.co/panini/](https://blog.granthika.co/panini/) 