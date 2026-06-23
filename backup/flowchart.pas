unit FlowChart;

Work Flow.
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
               |   MLP Dropout  |               |
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
               +----------------+               |
               |   R Dropout    |               |
               +----------------+               |
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


