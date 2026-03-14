unit Notes;

{$mode ObjFPC}{$H+}{$I proprietary.txt}

{ WesChat, Version 1.1, January 10, 2026, by Wesley R. Parsons, wespar@bellouth.net, www.wespar.com.}
{ ttest for git }
interface
implementation
begin
end.
{Notes

Vocabulary structure
These describe the symbol table itself and help you understand how your merges
shaped the final vocabulary.

Total vocabulary size — number of symbols.

Base (unmerged) symbols — usually 256 byte tokens.

Merged symbols — symbols with length > 1.

Merged-symbol length distribution — histogram of symbol lengths (e.g., how many 2‑byte merges, 3‑byte merges, etc.).

Average merged‑symbol length — mean length of all merged symbols.

Longest symbol length — maximum length of any merged token.

Vocabulary compression ratio — merged symbols ÷ total symbols.

These tell you whether your vocabulary is too shallow (few merges) or too bloated (many long merges that rarely appear).

Corpus tokenization statistics
These describe how the tokenizer actually behaved on the corpus.

Total token count — number of tokens in the tokenized corpus.

Merged token instances — count of tokens whose symbol length > 1.

Unmerged token instances — count of tokens whose symbol length = 1.

Merged‑instance ratio — merged instances ÷ total instances.

Average token length in characters — mean length of the symbol for each token instance.

Token length distribution — histogram of token lengths across the corpus.

Compression ratio (characters ÷ tokens) — how many characters per token on average.

This tells you how much compression your merges achieved and whether
the tokenizer is efficient on real text.

Frequency‑based insights
These help you understand which merges matter and which are dead weight.

Top N most frequent merged tokens — the merges doing the real work.

Merged tokens that never appear — candidates for pruning.

Merged tokens that appear only once — likely overfitting or noise.

Coverage of top merges — e.g., top 100 merges account for X% of merged instances.

This is where you can see whether your vocabulary is well-balanced or needs refinement.


To Do.

Fix computevocabfromlist.
In main program: Read Corpus, Read Files (vocab and merge), Tokenize, Embed, Transform.
One proc: display merge/token info. One proc: display transform/embed info. Move keypressed to global.

LoadCorpus    FileName          Corpus
LoadTables    Corpus            Symbol & Merge
Embed         Symbol & Merge    Sequence
Tokenize      Sequence          TokCorpus
RunForward    UserInput         Output

What to do with nTokens and append proc.
Check adding EOS and BOS at end of multiple corpuses.
Should SeqLen be a BVectorType.
In Tokenize, add maxheap with a hash table to speed up tokenization.
🔹 Store attention softmax outputs
Do I need them intact for backprop through softmax.

Resolved.
Use Welford addition. No, not with sgemm.
Use nSymbols in Tokenization, and nVocab in Training.
In tokenize, add longest token, and 20 most common tokens. No, too much trouble. Remove.
Put Hidden on the heap; make it a dynamically allocated variable.
  No. cblas will not work.

                        X
                       |||
              +------------------+
              |    Layer-Norm    |
              +------------------+
                       |||
                       X1
                       |||
              +------------------+
              |    Head Slice    |   Not done; reserve X0.
              +------------------+
                       |||
                       X1 >---------------------V
                       |||                      |
              +------------------+              |
              |     Attention    |              |
              +------------------+              |
                       |||                      |
               +----------------+               |
               | Split X1 Heads |               |
               +----------------+               |
                       |||                      |
               +----------------+               |
               |   Apply RoPE   |               |
               +----------------+               |
                       |||                      |
               +----------------+               |
               |   Wq, Wk, Wv   |               |
               +----------------+               |
                       |||                      |
                     Q, K, V                    |
                       |||                      |
               +----------------+               |
               |  Scores1=Q·Kt  |               |
               +----------------+               |
                       |||                      |
                     Scores1                    |
                       |||                      |
               +----------------+               |
               |  Standardize   |               |
               +----------------+               |
                       |||                      |
               +----------------+               |
               |     Masking    |               |
               +----------------+               |
                       |||                      |
                     Scores1                    |
                       |||                      |
               +----------------+               |
               |     Softmax    |               |
               +----------------+               |
                       |||                      |
                     Scores2                    |
                       |||                      |
               +----------------+               |
               |   A Dropout    |               |
               +----------------+               |
                       |||                      |
                     Scores2                    |
                       |||                      |
               +----------------+               |
               |  X2=Scores2·V  |               |
               +----------------+               |
                       |||                      |
               +----------------+               |
               |  Concat Heads  |               |
               +----------------+               |
                       |||                      |
                       X2                       |
                       |||                      |
              +------------------+              |
              |   Feed Forward   |              |
              +------------------+              |
                       |||                      |
                       X2                       |
                       |||                      |
              +------------------+              |
              |     X3=X2·W0     |              |
              +------------------+              |
                       |||                      |
                       X3                       |
                       |||                      |
              +------------------+              |
              |     X4=X3+X1     |<-------------<
              +------------------+
                       |||
                       X4
                       |||
              +------------------+
              |     Layer Norm   |
              +------------------+
                       |||
                       X5 >---------------------V
                       |||                      |
              +------------------+              |
              |     Activation   |              |
              +------------------+              |
                       |||                      |
               +----------------+               |
               |  H1=X5·W1+b1   |               |
               +----------------+               |
                       |||                      |
                     Hidden1                    |
                       |||                      |
               +----------------+               |
               |      ReLU      |               |
               +----------------+               |
                       |||                      |
                     Hidden2                    |
                       |||                      |
               +----------------+               |
               |   R Dropout    |               |
               +----------------+               |
                       |||                      |
                     Hidden2                    |
                       |||                      |
               +----------------+               |
               |  X6=H2·W2+b2   |               |
               +----------------+               |
                       |||                      |
                       X6                       |
                       |||                      |
              +------------------+              |
              |     X7=X6+X5     |<-------------<              |
              +------------------+
                       |||
                       X7
                       |||
              +------------------+
              |      Softmax     |
              +------------------+
                       |||
                      Logit
                       |||
              +------------------+
              | Gradient < Logit |
              +------------------+
                       |||
                   TopGradient

Program.
  Test.
  Tokenize file.
  Tokenize batch files.
  Input tokens.
  Tokenize. (optional)
  Embed, Sequence Loop.
    Init weights & biases.
    Loop thru blocks.
      Train.
        Init grads.
        Attention.
          Head split.
          Head concat.
        FFN.
        HeadOutput.
        LossFunction.
        BackPropopagate.
      ModifyWeights.
}
{   Corpus                 Extra
      V
  Symbol Table          Merge List
      V                 Meta Info
  Token List
      V
Print | Display
   TCorpus

Pipeline 1                   Pipeline 2

Corpus                       Corpus
  V                            V
  V                            V
Create Symbol Table          Read Symbol Table
  Read bytes                   Apply to Corpus
  Linked lists                    V
  Count pairs                     V
  Sort pairs                      V
  Merge Pairs                     V
  Convert to array                V
         V                        V
         V                        V
                TokenizedCorpus
                  V
                  V
                Stats
                  Create stats
                  Save stats


}
