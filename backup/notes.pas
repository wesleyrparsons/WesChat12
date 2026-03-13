unit Notes;

{$mode ObjFPC}{$H+}{$I proprietary.txt}

{ WesChat, Version 1.1, January 10, 2026, by Wesley R. Parsons, wespar@bellouth.net, www.wespar.com.}

interface
implementation
begin
end.
{Notes

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
