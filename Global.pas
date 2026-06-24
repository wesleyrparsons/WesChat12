unit Global;

{$mode ObjFPC}{$H+}{$I proprietary.txt}

{ WesChat, Version 1.2, begun January 10, 2026, by Wesley R. Parsons, wespar@bellouth.net, www.wesparsons.com.}

interface

uses Classes;

var
{ Place all verbosity and control options at start }
  DoNotPause: Boolean = False;              // Pause disabled.
  PauseIfKeyPressed: Boolean = True;        // Pause if a key is pressed.
  StopTraining: Boolean;                    // Happens if KeyPressed, to perform options.
  TrainSuccess: Boolean = False;            // Training successful and proceed to inference.
  DisplayCorpus: Boolean = False;            // One set for real tokenizing and one set for debug.
  DisplayWindow: Boolean = False;           // Display the SeqLen window.
  VerboseTokenize: Boolean = False;         // Verbose in Tokenize units.
  VerboseTransform: Boolean = False;        // Displays X, Q, ScoresHead1, etc. in Transform units.
  VeryVerboseTokenize: Boolean = False;     // Very verbose in Tokenize units.
  DisplayTokenWork: Boolean = False;            // Show token work in Tokenize units.
  DisplayMergeWork: Boolean = False;            // Show merge work in Tokenize units.
  DisplayVerification: Boolean = False;         // Do verification by rebuilding corpus in Tokenize units.
  DisplayEachByteRead: Boolean = False;        // Verify reading of bytes.
  SaveFiles: Boolean = True;                // Save various files, otherwise not saved.
  SaveTokenizationFiles: Boolean = True;    // Fave tokenization files (false for inference).
  MaxMerges: Integer = 20000;               // Maximum number of merges.
  MaxPairCount: Integer = 400000;           // Maximum number of pair in BPE.
  // SavePartialSymbolTable: Boolean = False;  // Save intermediate symbol tables.
  // PartialSymbolTableTrigger: Integer = 5000;// Trigger to save symbol tables.

const
  // Model constants.
  MaxEpochs = 100000;             // Number of epochs, loops over tokenized corpus,
  ModelDim = 64;                  // Number of loadings for a symbol.
  Proj = 4;                       // Projection to Hidden arrays.
  ModelDimProj = ModelDim * Proj; // Dimension of model of projected X matrix.
  SeqLen = 64;                    // Sequence length for X.
  Stride = 12;                    // Stride across sequence lengths.
  nHead = 4;                      // Number of heads for multi-headed attention.
  HeadDim = ModelDim div nHead;   // Length of one head.
  nBlock = 4;                     // Number of blocks in transformer.
  ADropOut = 0.1;                 // Probability of attention dropout.
  MLPDropOut = 0.1;               // Probability of MLP dropout.
  RDropout = 0.1;                 // Probability of residual dropout.
  DimVocab = 12000;                // Need maximum of vocab symbols to dimension array. Needed for Embeddings.
  // Transform constants.
  InvSqrtHeadDim: Single = 1 / Sqrt(HeadDim);         // Used in softmax.
  RowMajor = 101;                 // Row Major.
  NoTrans  = 111;                 // No transposition.
  Trans    = 112;                 // Transposition.


type                                                                           // SeqLen = L, ModelDim = D, ModelDim/nHead = H, DB is Proj*D, DV is DimVocab.
  // cublas type.
  TcublasHandle = Pointer;
  // Tokenizer type.
  TTokenizer = (WesTokenizer, GPT2Tokenizer);
  // Seq, Model, Vocab, Head, Proj types.
  TSeqVector = array [0..ModelDim - 1] of Single;                              // D
  TSeqVectorProj = array[0..ModelDimProj - 1] of Single;                       // DB (DB is like D)
  TDimVector = array[0..SeqLen - 1] of Single;                                 // L
  TIDimVector = array[0..SeqLen - 1] of Integer;                               // L
  THeadVector = array[0..HeadDim - 1] of Single;                               // H (H is like D)
  TVocabVector = array[0..DimVocab - 1] of Single;                             // DV (DV like L)
  TSeqMatrix = array[0..SeqLen - 1] of TSeqVector;                             // L x D
  TSeqHeadMatrix = array[0..SeqLen - 1] of THeadVector;                        // L x H
  TWeightMatrix = array[0..ModelDim - 1] of TSeqVector;                        // D x D
  TWeightHeadMatrix = array[0..HeadDim - 1] of THeadVector;                    // H x H        ?
  TWeightProjMatrix = array[0..ModelDim - 1] of TSeqVectorProj;                // D x DB
  TWeightProjMatrixT = array[0..ModelDimProj - 1] of TSeqVector;               // DB x D
  THiddenMatrix = array[0..SeqLen - 1] of TSeqVectorProj;                      // L x DB
  TScoresMatrix = array[0..SeqLen - 1] of TDimVector;                          // L x L
  TSeqVocabMatrix = array [0..SeqLen - 1] of TVocabVector;                     // L x MaxVocab
  TFSVector = array[0..SeqLen - 1] of Single;                                  // L
  TEmbeddingsMatrix = array[0..DimVocab - 1] of TSeqVector;                    // DV x D. Array for embeddings matrix, at DimVocab, a maximum.
  // Tensor types.
  TSeqTensor = record
    Value, Grad:  TSeqMatrix;
    dValue, dGrad:  PSingle;
  end;
  TSeqHeadTensor = record
    Value, Grad:  TSeqHeadMatrix;
    dValue, dGrad:  PSingle;
  end;
  TSeqVectorTensor = record
    Value, Grad:  TSeqVector;
    dValue, dGrad:  PSingle;
  end;
  THiddenTensor = record
    Value, Grad:  THiddenMatrix;
    dValue, dGrad:  PSingle;
  end;
  TSeqVectorProjTensor = record
    Value, Grad:  TSeqVectorProj;
    dValue, dGrad:  PSingle;
  end;
  TWeightTensor = record
    Value, Grad:  TWeightMatrix;
    dValue, dGrad:  PSingle;
  end;
  TWeightHeadTensor = record
    Value, Grad:  TWeightHeadMatrix;
    dValue, dGrad:  PSingle;
  end;
  TWeightProjTensor = record
    Value, Grad:  TWeightProjMatrix;
    dValue, dGrad:  PSingle;
  end;
  TWeightProjTensorT = record
    Value, Grad:  TWeightProjMatrixT;
    dValue, dGrad:  PSingle;
  end;
  TScoresHeadTensor = record
    Value, Grad:  TScoresMatrix;
    dValue, dGrad:  PSingle;
  end;
  TEmbeddingsTensor = record
    Value, Grad:  TEmbeddingsMatrix;
    dValue, dGrad:  PSingle;
  end;
  // Corpus and IO types.
  TBooleanVector = array of Boolean;   // Array of boolean.
  TIVector = array of Integer;         // Array of integers for corpuses.
  TBVector = array of Byte;            // Array of integers (UTF-8) for initial corpus.
  TRBSVector = array of RawByteString; // Array of raw byte strings for initial corpus.
  TSVector = array of String;          // Array of string.
  // Utility types.
  TFVector = array of Single;          // Array of single for RoPE.
  // Display types.
  TPart = (B, E, F, G);                // Length = VocabSize * Dimension. But only use nSymbols in rows.
  TSymbolTable = TRBSVector;           // Array of symbols. So index of array is a symbol string.
  // Block types.
  TParamBlock = array[0..nBlock - 1] of record
    Wq, Wk, Wv, W0:                 TWeightTensor;         // Weights.
    W1:                             TWeightProjTensor;     // Weights.
    W2:                             TWeightProjTensorT;    // Weights.
    b1:                             TSeqVectorProjTensor;  // Biases.
    b2:                             TSeqVectorTensor;      // Biases.
    Gamma1, Beta1, Gamma2, Beta2:   TSeqVectorTensor;      // Weights.
  end;
  TWModelParams = record                                   // Model of trainable parameters.
    Embeddings:                     TEmbeddingsTensor;     // Embeddings cannot be dynamic, CBLAS will not work.
    ParamBlock:                     TParamBlock;
  end;
  TStateBlock = array[0..nBlock - 1] of record                  // Model of non-trainable parameters.
    X, X1, X2, X3, X4, X5, X6, X7:  TSeqTensor;                 // X's at all stages.
    X1q, X1v, X1k:                  TSeqTensor;                 // X's for Q, K, V.
    Q, K, V:                        TSeqTensor;                 // Q is X*Wq, K is X*Wk, V is X*Wv.
    ScoresHead1, ScoresHead2:       array[0..nHead - 1] of TScoresHeadTensor;  // Scores partitioned into nHeads.
    Hidden1, Hidden2:               THiddenTensor;              // Neural net layer.
    // Caches for LayerNorm.
    LNInvStd1:  TFSVector;                                 // Cache for inverse standard deviation in LayerNorm.
    dLNInvStd1: PSingle;
    LNXhat1:    TSeqMatrix;                                // Cache for Xhat in LayerNorm.
    dLNXhat1:   PSingle;
    LNInvStd2:  TFSVector;                                 // Cache for inverse standard deviation in LayerNorm.
    dLNInvStd2: PSingle;
    LNXhat2:    TSeqMatrix;                                // Cache for Xhat in LayerNorm.
    dLNXhat2:   PSingle;
    // Dropout seeds.
    ADropoutSeed:         UInt64;                          // Seeds for dropouts.
    MLPDropoutSeed:       UInt64;
    RDropoutSeed:         UInt64;
    dX4FromLN2:           PSingle;
    dXFromLN1:            PSingle;
  end;
  TWModelState = record                                         // Model of non-trainable parameters.
    StateBlock:                     TStateBlock;
    InvFreq:                        TFVector;                   // Rope.
    dInvFreq:                       PSingle;
    Probs, TopGradient:             TSeqVocabMatrix;            // Logit and Gradient.
    dProbs, dTopGradient:           PSingle;
  end;

var
  // cublas vars.
  CuHandle: TcublasHandle;
  CudaAllocated: Boolean = False;
  // CuBlasInitialized: Boolean = False;
  One: Single = 1.0;
  Zero: Single = 0.0;
  // DLL accessibility vars.
  CublasPresent: Boolean;
  CudartPresent: Boolean;
  WesChatKernelPresent: Boolean;
  // Tokenize vars.                              // Need in order to use in decoding in Infer.
  Vocab: TStringList;
  // Corpus vars.
  CorpusFileNames: TSVector;                     // Name of corpus file.
  SymbolTable: TSymbolTable;                     // Symbol table.
  WorkingName, WorkingDir: string;               // Saving data.
  CorpusFileInfo: string;                        // Saving lon string of info on corpus.
  MultipleFileName: string;                      // Using multiple corpuses and outputting single file name.
  nTokenizedCorpus: Integer;                     // Length of tokenized corpus.
  // Target and Query vars.
  InputTokens: TIDimVector;                      // Input tokens.
  dInputTokens: PInteger;                        // Input tokens.
  TargetTokens: TIDimVector;                     // Target tokens. Shifted by  +1.
  dTargetTokens: PInteger;
  // Extra char vars.
  BOS: Integer = 256;                            // Begining of corpus.
  EOS: Integer = 257;                            // End of corpus.
  PAD: Integer = 258;                            // Padding to bring up to SeqLen.
  UNK: Integer = 259;                            // Unknown.
  // Model setings vars.
  BaseLearningRate:  Double = 0.01000;           // Base learning rate for Gradient.
  FloorLearningRate: Double = 0.00050;           // Floor learning rate for Gradient.
  OverrideLearningRate: Double = -1.0;           // Override learning rate for Gradient.
  RollOff:           Double = 0.99990;           // Reduction in learning rate.
  WeightDecay:       Double = 0.00010;           // Decay (multiplicative) for learning rate.
  LearningRate: Double;                          // Derived learningRate for Gradient.
  GlobalStep: Integer;                           // Increments once per window.
  DecayScale: Double;                            // 1.0 - LearningRate * WeightDecay.
  Temperature: Double = 1.0;                     // Temperature for softmax.
  ClipLimit: Double = 1.0;                       // Clips gradients.
  // Staging and epoch vars.
  DisplayStage: Boolean = False;                 // Display progress by stage in train and transform.
  DisplaySubstage: Boolean = False;              // Display progress by stage in train and transform.
  DisplayEpoch: Boolean = True;                  // Display progress by epoch in train and transform.
  Stage: Integer;                                // Indentation for stage;
  // Saving vars.
  WorkRoot: string = '';
  CorpusDir: string = '';
  SymbolDir: string = '';
  TokenDir: string = '';
  ModelDir: string = '';
  LogDir: string = '';
  RunDir: string = '';
  ScratchDir: string = '';
  // Utility vars.
  Mt0, Mt1, t0, t1, StopTime: TDateTime;         // For timing.
  Version: shortstring = '1.2';                  // Version 1.2.
  FromSymbolTable: Boolean = False;              // Operating from input Symbol Table rather than from tokenization.
  GlobalSeed: UInt64;                            // Initialize random sequence.
  nSymbols: Integer;                             // Number of symbols = Length(SymbolTable);
  nVocab: Integer;                               // nVocab is also nSymbol. Number of symbol items.
  TokenID: TIVector;                             // Same as TokenizedCorpus.
  Training: Boolean = True;                      // In training as opposed to inference mode.
  Tokenizer: TTokenizer;                         // WesChat or GPT2Chat tokenizer;
  NewModel: Boolean = True;                      // If new model, initialize params.
  ParamsNeedCopyToDevice: Boolean = False;       // Start of infer.
  CorpusPresent: Boolean = False;                 // Parts of program present;
  SymbolTablePresent: Boolean = False;
  MergeTablePresent: Boolean = False;
  TokenizedCorpusPresent: Boolean = False;
  ModelPresent: Boolean = False;
  QueryPresent: Boolean = False;

implementation

begin
end.

