unit Global;

{$mode ObjFPC}{$H+}{$I proprietary.txt}

{ WesChat, Version 1.2, begun January 10, 2026, by Wesley R. Parsons, wespar@bellouth.net, www.wespar.com.}

interface

var
{ Set the defaults }
  MaxMerges: Integer = 20000;          // Maximum number of merges.
  MaxPairCount: Integer = 400000;      // Maximum number of pair in BPE.

{ Place all verbosity and control options at start }
  DoNotPause: Boolean = False;         // Pause disabled.
  PauseIfKeyPressed: Boolean = True;   // Pause if a key is pressed.
  DisplayCorpus: Boolean = True;       // One set for real tokenizing and one set for debug.
  VeryVerbose: Boolean = False;
  VerboseTokenize: Boolean = True;
  VerboseTransform: Boolean = True;
  ShowTokenWork: Boolean = True;
  ShowMergeWork: Boolean = True;
  ShowVerification: Boolean = True;
  ShowEachByteRead: Boolean = False;
  SaveFiles: Boolean = True;
  SavePartialSymbolTable: Boolean = True;
  PartialSymbolTableTrigger: Integer = 5000;

const
  ModelDim = 160;                 // Number of loadings for a symbol.
  Proj = 4;                       // Projection to Hidden arrays.
  ModelDimProj = ModelDim * Proj; // Dimension of model of projected X matrix.
  SeqLen = 128;                   // Sequence length for X.
  nHead = 8;                      // Number of heads for multi-headed attention.
  HeadDim = ModelDim div nHead;   // Length of one head.
  nBlock = 4;                     // Number of blocks in transformer.
  ADropout = 0.1;                 // Probability of attention dropout.
  RDropout = 0.1;                 // Probability of residual dropout.
  DimVocab = 15000;               // Need maximum of vocab symbols to dimension array. Needed for Embeddings.

type                                                                           // SeqLen = L, ModelDim = D, ModelDim/nHead = H, DB is Proj*D
  TSeqVector = array [0..ModelDim - 1] of Single;                              // D
  TSeqVectorProj = array[0..ModelDimProj - 1] of Single;                       // DB (DB is like D)
  TDimVector = array[0..SeqLen - 1] of Single;                                 // L
  TIDimVector = array[0..SeqLen - 1] of Integer;                               // L
  THeadVector = array[0..HeadDim - 1] of Single;                               // H (H is like D)
  TVocabVector = array[0..DimVocab - 1] of Single;                             // MaxVocab (MaxVocab is like L)
  TSeqMatrix = array[0..SeqLen - 1] of TSeqVector;                             // L x D
  TSeqHeadMatrix = array[0..SeqLen - 1] of THeadVector;                        // L x H
  TWeightMatrix = array[0..ModelDim - 1] of TSeqVector;                        // D x D
  TWeightHeadMatrix = array[0..HeadDim - 1] of THeadVector;                    // H x H        ?
  TWeightProjMatrix = array[0..ModelDim - 1] of TSeqVectorProj;                // D x DB
  TWeightProjMatrixT = array[0..ModelDimProj - 1] of TSeqVector;               // DB x D
  THiddenMatrix = array[0..SeqLen - 1] of TSeqVectorProj;                      // L x DB
  TScoresMatrix = array[0..SeqLen - 1] of TDimVector;                          // L x L
  TVocabWeightMatrix = array[0..ModelDim - 1] of TVocabVector;                 // D x MaxVocab
  TSeqVocabMatrix = array [0..SeqLen - 1] of TVocabVector;                     // L x MaxVocab
  TFSVector = array[0..SeqLen - 1] of Single;                                  // L
  TEmbeddingsMatrix = array[0..DimVocab - 1] of TSeqVector;                    // Array for embeddings matrix.

  TSeqTensor = record
    Value, Grad:  TSeqMatrix;
  end;
  TSeqHeadTensor = record
    Value, Grad:  TSeqHeadMatrix;
  end;
  TSeqVectorTensor = record
    Value, Grad:  TSeqVector;
  end;
  THiddenTensor = record
    Value, Grad:  THiddenMatrix;
  end;
  TSeqVectorProjTensor = record
    Value, Grad:  TSeqVectorProj;
  end;
  TWeightTensor = record
    Value, Grad:  TWeightMatrix;
  end;
  TWeightHeadTensor = record
    Value, Grad:  TWeightHeadMatrix;
  end;
  TWeightProjTensor = record
    Value, Grad:  TWeightProjMatrix;
  end;
  TWeightProjTensorT = record
    Value, Grad:  TWeightProjMatrixT;
  end;
  TScoresHeadTensor = record
    Value, Grad:  TScoresMatrix;
  end;
  TEmbeddingsTensor = record
    Value, Grad:  TEmbeddingsMatrix;
  end;

  TBooleanVector = array of Boolean;   // Array of Boolean.
  TIVector = array of Integer;         // Array of integers for corpuses.
  TBVector = array of Byte;            // Array of integers (UTF-8) for initial corpus.
  TRBSVector = array of RawByteString; // Array of raw byte strings for initial corpus.
  TFVector = array of Single;          // Array of single for embedding matrix.
  TFMatrix = array of TFVector;        // Array of array of single for embedding matrix.
  TSVector = array of String;          // Array of string.
  TPart = (B, E, F, G);                // Length = VocabSize * Dimension. But only use nSymbols in rows.
  TSymbolTable = TRBSVector;           // Array of symbols. So index of array is a symbol string.
  TWModelParams = record               // Model of trainable parameters.
    Embeddings:                     TEmbeddingsTensor;     // Embeddings cannot be dynamic, CBLAS will not work.
    Wq, Wk, Wv, W0:                 TWeightTensor;         // Weights.
    W1:                             TWeightProjTensor;     // Weights.
    W2:                             TWeightProjTensorT;    // Weights.
    b1:                             TSeqVectorProjTensor;  // Biases.
    b2:                             TSeqVectorTensor;      // Biases.
    Gamma1, Beta1, Gamma2, Beta2:   TSeqVectorTensor;      // Weights.
  end;
  TWModelState = record                // Model of non-trainable parameters.
    X, X1, X2, X3, X4, X5, X6, X7:  TSeqTensor;              // X's at all stages.
    X1q, X1v, X1k:                  TSeqTensor;              // X's for Q, K, V.
    Q, K, V:                        TSeqTensor;              // Q is X*Wq, K is X*Wk, V is X*Wv.
    ScoresHead1, ScoresHead2:       array[0..nHead - 1] of TScoresHeadTensor;    // Scores partitioned into nHeads.
    Hidden1, Hidden2:               THiddenTensor;           // Neural net payer.
    Probs, TopGradient:             TSeqVocabMatrix;         // Logit and Gradient.
  end;

var
  // General.
  CorpusFileNames: TSVector;                     // Name of corpus file.
  SymbolTable: TSymbolTable;                     // Symbol table.
  WorkingName, WorkingDir: string;               // Saving data.
  CorpusFileInfo: string;                        // Saving lon string of info on corpus.
  nSymbols: Integer;                             // Number of symbols = Length(SymbolTable);
  nTokenizedCorpus: Integer;                     // Length of tokenized corpus.
  TokenID: TIVector;                             // Same as TokenizedCorpus.
  TargetTokens: TIDimVector;                     // Input and target tokenns. Input lags by one.
  nVocab: Integer = DimVocab;                    // nVocab is also nSymbol. Number of symbol items.
  Mt0, Mt1, t0, t1, StopTime: TDateTime;         // For timing.
  Version: shortstring = '1.2';                  // Version 1.2.
  FromSymbolTable: Boolean = False;              // Operating from input Symbol Table rather than from tokenization.
  MultipleFileName: string;                      // Using multiple corpuses and outputting single file name.
  XSize: Integer = SeqLen * ModelDim;            // Size of X matrices.
  HiddenSize: Integer = SeqLen * ModelDimProj;   // Size of Hidden matrices.
  LearningRate: Single = 0.01;                   // LearningRate for Gradient.
  Temperature: Single = 1.0;                     // Temperature for softmax.
  Training: Boolean = True;                      // In training as opposed to inference mode.

  // Non-trainable parameters.
  WModelState: TWModelState;

  // Caches.
  LNInvStd1:  TFSVector;          // Caches for Layer-Norm.
  LNXhat1:    TSeqMatrix;         // Caches for Layer-Norm.
  LNInvStd2:  TFSVector;          // Caches for Layer-Norm.
  LNXhat2:    TSeqMatrix;         // Caches for Layer-Norm.

  // Other.
  TestVector: TFSVector;          // Vector for testing. [0..SeqLen] of Single.
  InvFreq:    TFVector;           // For RoPE.

implementation

begin

end.

