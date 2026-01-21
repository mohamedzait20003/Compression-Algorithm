# LLM-Based Text Compression

Lossless text compression using **LLM-based arithmetic coding** with Llama 3.2 1B model.

## Algorithm

This compressor uses a neural language model to predict the probability distribution of the next token, then applies **arithmetic coding** to encode the text with near-optimal bit efficiency.

### How It Works

```
┌─────────────────────────────────────────────────────────────────┐
│                    COMPRESSION PIPELINE                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input Text ──► Tokenize ──► LLM Prediction ──► Arithmetic     │
│                                    │            Coding          │
│                                    ▼               │            │
│                            P(next_token|context)   ▼            │
│                                                 Binary          │
│                                                 Output          │
└─────────────────────────────────────────────────────────────────┘
```

1. **Tokenization**: Text is split into tokens using the LLM's tokenizer
2. **Probability Estimation**: For each token, the LLM predicts P(token | previous_context)
3. **Arithmetic Coding**: Tokens are encoded using their predicted probabilities
   - High probability tokens → fewer bits
   - Low probability tokens → more bits
4. **Binary Output**: Final compressed stream as bytes

### Why LLM + Arithmetic Coding?

| Method | Bits/Word | Limitation |
|--------|-----------|------------|
| ASCII | 40-50 | No compression |
| Huffman | 15-20 | Fixed probabilities |
| Word-level Arithmetic | 8-10 | No context |
| **LLM + Arithmetic** | **7-9** | Context-aware prediction |

The LLM exploits **word-to-word dependencies** that static methods cannot capture:
- "How are ___" → "you" is highly predictable → few bits
- Rare word after common context → more bits

## Results

Tested on **1,000 sentences** from the DailyDialog dataset:

| Metric | Value |
|--------|-------|
| Messages tested | 1,000 |
| Total original | 61,973 bytes |
| Total compressed | 13,098 bytes |
| **Compression ratio** | **4.73x** |
| **Space savings** | **78.9%** |
| **Avg bits/word** | **9.04** |
| Avg bits/token | 8.54 |
| Verification | 100% ✓ |

## Detailed Results

Here are 100 compression results from the DailyDialog dataset:

| # | Message (truncated) | Original | Compressed | Ratio | Bits/Word |
|---|---------------------|----------|------------|-------|-----------|
| 1 | Say , Jim , how about going for a few beers after ... | 58 bytes | 13 bytes | 4.46x | 7.43 |
| 2 | You know that is tempting but is really not good f... | 66 bytes | 15 bytes | 4.40x | 8.57 |
| 3 | What do you mean ? It will help us to relax . | 45 bytes | 12 bytes | 3.75x | 8.00 |
| 4 | Do you really think so ? I don't . It will just ma... | 96 bytes | 22 bytes | 4.36x | 7.65 |
| 5 | I guess you are right.But what shall we do ? I don... | 80 bytes | 17 bytes | 4.71x | 7.56 |
| 6 | I suggest a walk over to the gym where we can play... | 90 bytes | 18 bytes | 5.00x | 7.20 |
| 7 | That's a good idea . I hear Mary and Sally often g... | 117 bytes | 25 bytes | 4.68x | 8.33 |
| 8 | Sounds great to me ! If they are willing , we coul... | 125 bytes | 24 bytes | 5.21x | 7.11 |
| 9 | Good.Let ' s go now . | 21 bytes | 11 bytes | 1.91x | 14.67 |
| 10 | All right . | 11 bytes | 6 bytes | 1.83x | 16.00 |
| 11 | Can you do push-ups ? | 21 bytes | 8 bytes | 2.62x | 12.80 |
| 12 | Of course I can . It's a piece of cake ! Believe i... | 92 bytes | 17 bytes | 5.41x | 5.67 |
| 13 | Really ? I think that's impossible ! | 36 bytes | 9 bytes | 4.00x | 10.29 |
| 14 | You mean 30 push-ups ? | 22 bytes | 9 bytes | 2.44x | 14.40 |
| 15 | Yeah ! | 6 bytes | 6 bytes | 1.00x | 24.00 |
| 16 | It's easy . If you do exercise everyday , you can ... | 65 bytes | 15 bytes | 4.33x | 7.50 |
| 17 | Can you study with the radio on ? | 33 bytes | 10 bytes | 3.30x | 10.00 |
| 18 | No , I listen to background music . | 35 bytes | 9 bytes | 3.89x | 9.00 |
| 19 | What is the difference ? | 24 bytes | 7 bytes | 3.43x | 11.20 |
| 20 | The radio has too many comerials . | 34 bytes | 12 bytes | 2.83x | 13.71 |
| 21 | That's true , but then you have to buy a record pl... | 56 bytes | 12 bytes | 4.67x | 7.38 |
| 22 | Are you all right ? | 19 bytes | 6 bytes | 3.17x | 9.60 |
| 23 | I will be all right soon . I was terrified when I ... | 83 bytes | 17 bytes | 4.88x | 7.16 |
| 24 | Don't worry.He is an acrobat 。 | 32 bytes | 12 bytes | 2.67x | 16.00 |
| 25 | I see . | 7 bytes | 6 bytes | 1.17x | 16.00 |
| 26 | Hey John , nice skates . Are they new ? | 39 bytes | 11 bytes | 3.55x | 8.80 |
| 27 | Yeah , I just got them . I started playing ice hoc... | 116 bytes | 20 bytes | 5.80x | 6.15 |
| 28 | What position do you play ? | 27 bytes | 8 bytes | 3.38x | 10.67 |
| 29 | I ' m a defender . It ' s a lot of fun . You don '... | 104 bytes | 21 bytes | 4.95x | 5.79 |
| 30 | Yeah , you ' re a pretty big guy . I play goalie ,... | 61 bytes | 15 bytes | 4.07x | 7.50 |
| 31 | Oh , yeah ? Which team ? | 24 bytes | 10 bytes | 2.40x | 11.43 |
| 32 | The Rockets . | 13 bytes | 8 bytes | 1.62x | 21.33 |
| 33 | Really ? I think we play you guys next week . Well... | 95 bytes | 17 bytes | 5.59x | 5.67 |
| 34 | All right , see you later . | 27 bytes | 8 bytes | 3.38x | 9.14 |
| 35 | Hey Lydia , what are you reading ? | 34 bytes | 10 bytes | 3.40x | 10.00 |
| 36 | I ' m looking at my horoscope for this month ! My ... | 187 bytes | 27 bytes | 6.93x | 5.54 |
| 37 | What are you talking about ? Let me see that ... W... | 70 bytes | 16 bytes | 4.38x | 8.53 |
| 38 | It ' s a prediction of your month , based on your ... | 203 bytes | 27 bytes | 7.52x | 4.41 |
| 39 | January 5th . | 13 bytes | 7 bytes | 1.86x | 18.67 |
| 40 | Let ' s see . . . you ' re a Capricorn . It says t... | 213 bytes | 36 bytes | 5.92x | 6.13 |
| 41 | That ' s bogus . I don't feel any stress at work ,... | 145 bytes | 23 bytes | 6.30x | 5.94 |
| 42 | No , it ' s not , your astrology sign can tell you... | 149 bytes | 24 bytes | 6.21x | 5.82 |
| 43 | Well , you certainly match those criteria , but th... | 125 bytes | 21 bytes | 5.95x | 6.22 |
| 44 | A Capricorn is serious-minded and practical . She ... | 119 bytes | 20 bytes | 5.95x | 7.27 |
| 45 | Frank ' s getting married , do you believe this ? | 51 bytes | 13 bytes | 3.92x | 9.45 |
| 46 | Is he really ? | 14 bytes | 7 bytes | 2.00x | 14.00 |
| 47 | Yes , he is . He loves the girl very much . | 43 bytes | 11 bytes | 3.91x | 7.33 |
| 48 | Who is he marring ? | 19 bytes | 10 bytes | 1.90x | 16.00 |
| 49 | A girl he met on holiday in Spain , I think . | 45 bytes | 13 bytes | 3.46x | 8.67 |
| 50 | Have they set a date for the wedding ? | 38 bytes | 8 bytes | 4.75x | 7.11 |
| 51 | Not yet . | 9 bytes | 6 bytes | 1.50x | 16.00 |
| 52 | I hear you bought a new house in the northern subu... | 55 bytes | 12 bytes | 4.58x | 8.00 |
| 53 | That ' s right , we bought it the same day we came... | 68 bytes | 14 bytes | 4.86x | 6.59 |
| 54 | What kind of house is it ? | 26 bytes | 7 bytes | 3.71x | 8.00 |
| 55 | It ' s a wonderful Spanish style . | 36 bytes | 11 bytes | 3.27x | 11.00 |
| 56 | Oh , I love the roof tiles on Spanish style houses... | 52 bytes | 13 bytes | 4.00x | 8.67 |
| 57 | And it ' s a bargaining . A house like this in riv... | 84 bytes | 21 bytes | 4.00x | 8.84 |
| 58 | Great , is it a two bedroom house ? | 35 bytes | 10 bytes | 3.50x | 8.89 |
| 59 | No , it has three bedrooms and three beds , and ha... | 125 bytes | 19 bytes | 6.58x | 5.63 |
| 60 | That ' s a nice area too . It ' ll be a good inves... | 69 bytes | 14 bytes | 4.93x | 6.22 |
| 61 | Yeas , when will you buy a house ? | 34 bytes | 12 bytes | 2.83x | 10.67 |
| 62 | Not untill the end of this year , you know , just ... | 69 bytes | 15 bytes | 4.60x | 7.50 |
| 63 | Right , congratulations . | 25 bytes | 8 bytes | 3.12x | 16.00 |
| 64 | Thank you . | 11 bytes | 6 bytes | 1.83x | 16.00 |
| 65 | Hi , Becky , what's up ? | 24 bytes | 9 bytes | 2.67x | 10.29 |
| 66 | Not much , except that my mother-in-law is driving... | 67 bytes | 12 bytes | 5.58x | 6.86 |
| 67 | What's the problem ? | 20 bytes | 6 bytes | 3.33x | 12.00 |
| 68 | She loves to nit-pick and criticizes everything th... | 109 bytes | 17 bytes | 6.41x | 6.48 |
| 69 | For example ? | 13 bytes | 7 bytes | 1.86x | 18.67 |
| 70 | Well , last week I invited her over to dinner . My... | 229 bytes | 36 bytes | 6.36x | 5.76 |
| 71 | No , I can't see that happening . I know you're a ... | 101 bytes | 16 bytes | 6.31x | 5.82 |
| 72 | It's not just that . She also criticizes how we ra... | 64 bytes | 14 bytes | 4.57x | 8.00 |
| 73 | My mother-in-law used to do the same thing to us .... | 248 bytes | 31 bytes | 8.00x | 4.96 |
| 74 | You said she used to ? How did you stop her ? | 45 bytes | 12 bytes | 3.75x | 8.00 |
| 75 | We basically sat her down and told her how we felt... | 214 bytes | 29 bytes | 7.38x | 5.40 |
| 76 | That sounds like a good idea . I'll have to try th... | 54 bytes | 9 bytes | 6.00x | 5.54 |
| 77 | How are Zina's new programmers working out ? | 44 bytes | 13 bytes | 3.38x | 13.00 |
| 78 | I hate to admit it , but they're good . And fast .... | 81 bytes | 18 bytes | 4.50x | 7.20 |
| 79 | So you'll make the Stars.com deadline , and have u... | 78 bytes | 19 bytes | 4.11x | 9.50 |
| 80 | It'll be close , but we'll make it . | 36 bytes | 9 bytes | 4.00x | 8.00 |
| 81 | Good . After Stars.com starts paying us , we won't... | 78 bytes | 23 bytes | 3.39x | 12.27 |
| 82 | And if we don't need them , we won't need Zina , e... | 57 bytes | 14 bytes | 4.07x | 8.00 |
| 83 | Do you like cooking ? | 21 bytes | 7 bytes | 3.00x | 11.20 |
| 84 | Yes . I like cooking very much . I got this hobby ... | 76 bytes | 16 bytes | 4.75x | 6.74 |
| 85 | Why do you like it ? | 20 bytes | 7 bytes | 2.86x | 9.33 |
| 86 | I have no idea . I like cooking by myself . I like... | 76 bytes | 15 bytes | 5.07x | 6.67 |
| 87 | That's wonderful ! | 18 bytes | 6 bytes | 3.00x | 16.00 |
| 88 | And I love trying new recipes , which I usually te... | 91 bytes | 17 bytes | 5.35x | 6.48 |
| 89 | Really ? I hope I can have a chance to taste it . ... | 75 bytes | 15 bytes | 5.00x | 6.32 |
| 90 | Certainly . | 11 bytes | 6 bytes | 1.83x | 24.00 |
| 91 | Anyone home ? Jen ! | 19 bytes | 9 bytes | 2.11x | 14.40 |
| 92 | I'm in the kitchen ... let yourself in ! | 40 bytes | 12 bytes | 3.33x | 10.67 |
| 93 | Wow ! You're really working up a storm ! | 40 bytes | 10 bytes | 4.00x | 8.89 |
| 94 | I know . I've even worked up a sweat . | 38 bytes | 12 bytes | 3.17x | 9.60 |
| 95 | You look like a cooking show host--only messier . | 49 bytes | 14 bytes | 3.50x | 12.44 |
| 96 | You look so tan and healthy ! | 29 bytes | 8 bytes | 3.62x | 9.14 |
| 97 | Thanks . I just got back from summer camp . | 43 bytes | 9 bytes | 4.78x | 7.20 |
| 98 | How was it ? | 12 bytes | 6 bytes | 2.00x | 12.00 |
| 99 | Great . I got to try so many things for the first ... | 56 bytes | 11 bytes | 5.09x | 6.29 |
| 100 | Like what ? | 11 bytes | 6 bytes | 1.83x | 24.00 |

## Installation

### Prerequisites

**Windows:**
- Python 3.10+
- C/C++ Build Tools (required for llama-cpp-python)
  - Install [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
  - Or install Visual Studio with "Desktop development with C++" workload

**Linux:**
```bash
sudo apt-get update
sudo apt-get install build-essential gcc g++ cmake python3-dev
```

**macOS:**
```bash
xcode-select --install
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

The model (~770MB) will be downloaded automatically on first run.

## Usage

### Compress a single sentence
```bash
python main.py -t "Hello, how are you doing today?"
```

### Compress from file (one sentence per line)
```bash
python main.py -f sentences.txt
```

### Output Example
```
======================================================================
INPUT: "Hello, how are you doing today?"
======================================================================

COMPRESSED BINARY (72 bits):
  0110101011001110101011110011001010110101001110101011110011001010

STATISTICS:
  Original:              31 bytes (   248 bits)
  Compressed:             9 bytes (    72 bits)
  Compression ratio:   3.44x
  Space savings:        71.0%
  Bits per word:        12.00
  Bits per token:        9.00
  Verified:          ✓ Yes
```

## Project Structure

```
├── main.py              # CLI compressor
├── requirements.txt     # Dependencies
├── results.txt          # Full test results (1000 sentences)
├── llama-zip/           # LLM compression library
└── models/
```

## Requirements

- Python 3.10+
- ~1GB disk space for model
- 4GB+ RAM recommended
