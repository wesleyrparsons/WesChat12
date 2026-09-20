unit Global;

{$mode ObjFPC}{$H+}{$I proprietary.txt}

{ WesChat, Version 1.2, begun January 10, 2026, by Wesley R. Parsons, wespar@bellsouth.net, www.wesparsons.com.}

interface

uses Classes;

var
{ Place all verbosity and control options at start }
  // Pausing.
  DoNotPause: Boolean = False;                   // Pause disabled.
  PauseIfKeyPressed: Boolean = True;             // Pause if a key is pressed.
  StopTraining: Boolean;                         // Happens if KeyPressed, to perform options.
  TrainSuccess: Boolean = False;                 // Training successful and proceed to inference.
  // Verbosity.
  VerboseTokenize: Boolean = False;              // Display steps in Tokenize and Symbolize units.
  VeryVerboseTokenize: Boolean = False;          // Very verbose in Tokenize units.
  VerboseTransform: Boolean = False;             // Displays X, Q, ScoresHead1, etc. in Transform units.
  VerboseInfer: Boolean = False;                 // Displays many steps in Infer unit.
  DetailInfer: Boolean = False;                  // Displays some steps in Infer unit.
  // Displaying data.
  DisplayCorpus: Boolean = False;                // One set for real tokenizing and one set for debug.
  DisplayWindow: Boolean = False;                // Display the SeqLen window.
  DisplayTokenWork: Boolean = False;             // Show token work in Tokenize units.
  DisplayMergeWork: Boolean = False;             // Show merge work in Tokenize units.
  DisplayCorpusVerification: Boolean = True;     // Verify by rebuilding corpus in WesTokenize or displaying in GPT.
  DisplayTokenVerification: Boolean = False;     // Verify by rebuilding tokens in WesTokenize or displaying in GPT.
  DisplayEachByteRead: Boolean = False;          // Verify reading of bytes.
  // Saving.
  SaveFiles: Boolean = True;                     // Save various files, otherwise not saved.
  SaveTokenizationFiles: Boolean = True;         // Save tokenization files (false for inference).
  // Pairs and merging.
  MaxMerges: Integer = 60000;                    // Maximum number of merges.
  MaxPairCount: Integer = 800000;                // Maximum number of pair in BPE.

const
  // Model constants.
  MaxEpochs = 1000000;            // Number of epochs, loops over tokenized corpus.
  ModelDim = 192;                 // Number of loadings for a symbol.
  Scale = Sqrt(ModelDim);         // Transformer-style embedding scaling by sqrt(d_model).
  Proj = 4;                       // Projection to Hidden arrays.
  ModelDimProj = ModelDim * Proj; // Dimension of model of projected X matrix.
  SeqLen = 256;                   // Sequence length for X.
  nHead = 8;                      // Number of heads for multi-headed attention.
  HeadDim = ModelDim div nHead;   // Length of one head.
  nBlock = 6;                     // Number of blocks in transformer.
  MaxSymbols = 50260;             // Maximum WesTokenizer symbol count during BPE construction.
  DimVocab   = 50260;             // Physical model vocabulary capacity; must be >= nVocab. Use 50260 for GPT2.
  // GPT2 constants.
  GPT2BaseVocabSize = 50257;
  GPT2EOS = 50256;                // Official GPT-2 end-of-text token.
  GPT2PAD = 50257;                // WesChat extension.
  GPT2BOS = 50258;                // WesChat extension.
  GPT2UNK = 50259;                // WesChat extension; normally never needed.
  GPT2ModelVocabSize = 50260;
  ModelMagic:  array[0..7] of Char = ('W','E','S','2','M','O','D','L');
  // Other constants.
  RecentCount = 10;               // Rolling means in training.
  Version: shortstring = '1.2';   // Version 1.2.
  FirstMergedToken = 400;         // First token after all extended ASCII and 4 specials.
  DisplayLength = 100;            // Length of diaplying corpus or tokens.
  InvSqrtHeadDim: Single = 1 / Sqrt(HeadDim);    // Used in softmax.

type                              // SeqLen = L, ModelDim = D, ModelDim/nHead = H, DB is Proj*D, DV is DimVocab.
  // Cublas type.
  TcublasHandle = Pointer;
  // Tokenizer kind type.
  TTokenizerKind = (WesTokenizer, UDTokenizer, GPT2Tokenizer);
  // Seq, Model, Vocab, Head, Hidden, Scores, Embedding, Proj types.
  TSeqVector = array [0..ModelDim - 1] of Single;                              // D
  TSeqVectorProj = array[0..ModelDimProj - 1] of Single;                       // DB (DB is like D)
  TDimVector = array[0..SeqLen - 1] of Single;                                 // L
  TIDimVector = array[0..SeqLen - 1] of Integer;                               // L
  THeadVector = array[0..HeadDim - 1] of Single;                               // H (H is like D)
  TVocabVector = array[0..DimVocab - 1] of Single;                             // DV (DV like L)
  TSeqMatrix = array[0..SeqLen - 1] of TSeqVector;                             // L x D
  TWeightMatrix = array[0..ModelDim - 1] of TSeqVector;                        // D x D
  TWeightProjMatrix = array[0..ModelDim - 1] of TSeqVectorProj;                // D x DB
  TWeightProjMatrixT = array[0..ModelDimProj - 1] of TSeqVector;               // DB x D
  THiddenMatrix = array[0..SeqLen - 1] of TSeqVectorProj;                      // L x DB
  TScoresMatrix = array[0..SeqLen - 1] of TDimVector;                          // L x L
  TSeqVocabMatrix = array [0..SeqLen - 1] of TVocabVector;                     // L x MaxVocab
  TFSVector = array[0..SeqLen - 1] of Single;                                  // L
  TEmbeddingsMatrix = array[0..DimVocab - 1] of TSeqVector;                    // DV x D. Array for embeddings matrix, at DimVocab, a maximum.
  // State tensor types.
  TSeqTensor = record
    Value, Grad:  TSeqMatrix;
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
  TAdamWeightTensor = record
    M, V: TWeightMatrix;
    dM, dV: PSingle;
  end;
  // Adam state tensor types.
  TAdamWeightProjTensor = record
    M, V: TWeightProjMatrix;
    dM, dV: PSingle;
  end;
  TAdamWeightProjTensorT = record
    M, V: TWeightProjMatrixT;
    dM, dV: PSingle;
  end;
  TAdamSeqVectorTensor = record
    M, V: TSeqVector;
    dM, dV: PSingle;
  end;
  TAdamSeqVectorProjTensor = record
    M, V: TSeqVectorProj;
    dM, dV: PSingle;
  end;
  TAdamEmbeddingsTensor = record
    M, V: TEmbeddingsMatrix;
    dM, dV: PSingle;
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
  // Learning style types.
  TLearningStyle = (FlatLearning, FastLearning, SlowLearning, RolledOffLearning);
  // Adaptive LR.
  type
  TAdaptiveLRState = record
    Initialized: Boolean;
    PrevLoss: Double;
    PrevParamRMS: Double;
    PrevUpdateRatio: Double;
    PrevMRMS: Double;
    PrevSqrtVRMS: Double;
    PrevMaxGammaRMS: Double;
    ConsecutiveWorse: Integer;
    ConsecutiveFlat: Integer;
    LastLRChangeEpoch: Integer;
  end;
  type
    TTrainingCheckpoint = record
      GlobalStep: Int64;
      CompletedEpochs: Integer;
      LearningStyle: TLearningStyle;
      LearningRate: Double;
      OverrideLearningRate: Double;
      BaseLearningRate: Double;
      FloorLearningRate: Double;
      RollOff: Double;
      WeightDecay: Double;
      ClipLimit: Double;
      TTemperature: Double;
      ITemperature: Double;
      ADropOut: Double;
      RDropOut: Double;
      MLPDropOut: Double;
      ShuffleWindows: Boolean;
      Stride: Integer;
      StartStride: Integer;
      GlobalSeed: QWord;
      AdamWStep: Int64;
      AdamBeta1: Double;
      AdamBeta2: Double;
      AdamEpsilon: Double;
      AdaptiveLR: Boolean;
      AdaptiveLRState: TAdaptiveLRState;
      MinLoss: Double;
      MinLossEpoch: Integer;
      BestSavedLoss: Double;
      LastBestSaveEpoch: Integer;
    end;

  // Param block types.
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
  // State block types.
  TStateBlock = array[0..nBlock - 1] of record             // Model of non-trainable parameters.
    // Matrices for neural net.
    X, X1, X2, X3, X4, X5, X6, X7:  TSeqTensor;            // X's at all stages.
    X1q, X1v, X1k:                  TSeqTensor;            // X's for Q, K, V. Actaully, don't need dValue. Can simplify.
    Q, K, V:                        TSeqTensor;            // Q is X*Wq, K is X*Wk, V is X*Wv.
    ScoresHead1, ScoresHead2:       array[0..nHead - 1] of TScoresHeadTensor;  // Scores partitioned into nHeads.
    Hidden1, Hidden2:               THiddenTensor;         // Neural net layer.
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
    ADropoutSeed:         UInt64;                          // Dropout seeds.
    MLPDropoutSeed:       UInt64;
    RDropoutSeed:         UInt64;
    // Gradients for backprop.
    dX4FromLN2:           PSingle;
    dXFromLN1:            PSingle;
  end;
  TWModelState = record                                    // Model of non-trainable parameters.
    StateBlock:                     TStateBlock;           // See above.
    InvFreq:                        TFVector;              // Inverse frequency, for RoPE.
    dInvFreq:                       PSingle;               // Inverse frequency, for RoPE.
    Probs, TopGradient:             TSeqVocabMatrix;       // Logit and Gradient.
    dProbs, dTopGradient:           PSingle;               // Logit and Gradient.
    dRowLoss:                       PSingle;               // For cross-entropy loss.
  end;
  // Adam param abd state block types.
  TAdamParamBlock = array[0..nBlock - 1] of record
    Wq, Wk, Wv, W0: TAdamWeightTensor;
    W1: TAdamWeightProjTensor;
    W2: TAdamWeightProjTensorT;
    b1: TAdamSeqVectorProjTensor;
    b2: TAdamSeqVectorTensor;
    Gamma1, Beta1, Gamma2, Beta2: TAdamSeqVectorTensor;
  end;
  TWAdamWState = record
    Embeddings: TAdamEmbeddingsTensor;
    ParamBlock: TAdamParamBlock;
  end;
var
  // cublas vars.
  CuHandle: TcublasHandle;                       // Create a cuda handle.
  CudaAllocated: Boolean = False;                // Is cuda allocated?
  DebugCudaChecks: Boolean = False;              // Do checks on cuda -- this will slow execution.
  // DLL accessibility vars.
  CublasPresent: Boolean;                        // Is cublas present?
  CudartPresent: Boolean;                        // Is cudart present?
  WesChatKernelPresent: Boolean;                 // Is my kernel present?
  // Tokenize vars.                              // Need in order to use in decoding in Infer.
  // UDTagOn: Boolean = True;                       // True means use UD tags.
  UDPipeFileName: string = 'c:\wc\ud\udpipe.exe';
  UDModelFileName: string = 'c:\wc\ud\english-ewt-ud-2.4-190531.udpipe';
  Vocab: TStringList;                            // Vocabulary for ChatGPT.
  // Current work and file names.
  ExistingWorkRoot: string = 'C:\wc\';
  CurrentBaseName: string = 'weschat';
  CorpusFileNames: array of string;
  CorpusFileName: string = '';
  TokenFileName: string = '';
  SymbolFileName: string = '';
  VocabFileName: string = '';
  MergeFileName: string = '';
  ModelFileName: string = '';
  BestModelFileName: string = '';
  RunFileName: string = '';
  ListFile: string = '';
  LogFileName: string = '';
  // Corpus vars.
  nCorpus: Integer;                              // Length of corpus.
  MergeCount: Integer;                           // Count of merges.
  nMerges: Integer;                               // Number of merges.
  CorpusID: UInt64;                              // Identifier for corpus.
  SymbolTable: TSymbolTable;                     // Symbol table.
  WorkingName, WorkingDir: string;               // Saving data.
  CorpusFileInfo: string;                        // Saving long string of info on corpus.
  MultipleFileName: string;                      // Using multiple corpuses and outputting single file name.
  nTokenizedCorpus: Integer;                     // Length of tokenized corpus.
  RawTokenCount, PaddedTokenCount: Integer;      // Raw and padded token count.
  // Target and Query vars.
  InputTokens: TIDimVector;                      // Input tokens.
  dInputTokens: PInteger;                        // Input tokens.
  TargetTokens: TIDimVector;                     // Target tokens. Shifted by  +1.
  dTargetTokens: PInteger;                       // Target tokens.
  // Extra char vars.
  BOS: Integer = 256;                            // Begining of corpus.
  EOS: Integer = 257;                            // End of corpus.
  PAD: Integer = 258;                            // Padding to bring up to SeqLen.
  UNK: Integer = 259;                            // Unknown.
  // Model settings vars.
  Training:             Boolean = False;         // True = training mode: training temperature and dropout enabled.
  LearningStyle:        TLearningStyle = SlowLearning;     // Style of learning.
  TrainLogFile:         Text;                    // Log for training data.
  TrainLogOpen:         Boolean = False;         // Training log open?
  TrainLogFileName:     string = '';             // Name of training log.
  ShuffleWindows:       Boolean = True;          // Shuffle the windows each epoch.
  BaseLearningRate:     Double = 0.000100;       // Base learning rate for Gradient.
  FloorLearningRate:    Double = 0.000005;       // Floor learning rate for Gradient.
  OverrideLearningRate: Double = -1.00000;       // Override learning rate for Gradient.
  RollOff:              Double = 0.999900;       // Reduction in learning rate.
  WeightDecay:          Double = 0.000100;       // Decay (multiplicative) for learning rate.
  DecayScale:           Double;                  // 1.0 - LearningRate * WeightDecay.
  LearningRate:         Double;                  // Derived learningRate for Gradient.
  GlobalStep:           Int64;                   // Increments once per window.
  TTemperature:         Single = 1.00000;        // Training temperature for softmax.
  ITemperature:         Single = 1.00000;        // Inference temperature for softmax.
  ClipLimit:            Single = 0.40000;        // Clips gradients.
  CompletedEpochs:      Integer;                 // Number of epochs completed, for saving model.
  // Dropout probabilities and seed.
  ADropOut:             Single = 0.05;           // Probability of attention dropout. Set to 0.0 for no dropout.
  MLPDropOut:           Single = 0.05;           // Probability of MLP dropout.       Even if Training is True.
  RDropout:             Single = 0.05;           // Probability of residual dropout.
  GlobalSeed:           UInt64 = 123456789;      // Global seed.
  Stride:               Integer = 128;           // Stride across sequence lengths.
  StartStride:          Integer = 17;            // Coprime with Stride. Also use 43. Ty, Leonhard.
  // Staging and epoch vars.
  DisplayStage:         Boolean = False;         // Display progress by stage in train and transform.
  DisplaySubstage:      Boolean = False;         // Display progress by stage in train and transform.
  DisplayEpoch:         Boolean = True;          // Display progress by epoch in train and transform.
  Stage:                Byte;                    // Indentation for stage;
  // Adaptive LR adnd Loss.
  AdaptiveLR: Boolean = True;
  AdaptiveLRState: TAdaptiveLRState;
  MinLoss: Double;
  MinLossEpoch: Integer;
  BestSavedLoss: Double;
  LastBestSaveEpoch: Integer;
  // Saving vars.
  WorkRoot:             string = '';             // Work folder name set by user or default.
  CorpusDir:            string = '';             // Folder for corpus files.
  SymbolDir:            string = '';             // Folder for symbol files.
  MergeDir:             string = '';             // Folder for merge files.
  TokenDir:             string = '';             // Folder for token files.
  ModelDir:             string = '';             // Folder for model files.
  LogDir:               string = '';             // Folder for log files.
  RunDir:               string = '';             // Folder for run files (not used).
  ListDir:              string = '';             // Folder for list files (not used).
  ScratchDir:           string = '';             // Folder for scratch files (not used).
  // Adam hyperparameters.
  AdamBeta1:            Single = 0.90000;
  AdamBeta2:            Single = 0.99900;
  AdamEpsilon:          Single = 1.0e-8;
  AdamWStep:            Int64 = 0;
  AdamWStateLoaded:     Boolean = False;
  // Utility vars.
  Mt0, Mt1, t0, t1, StopTime: TDateTime;         // For timing.
  FromSymbolTable: Boolean = False;              // Operating from input Symbol Table rather than from tokenization.
  nSymbols: Integer;                             // Number of symbols produced/loaded by tokenizer.
  nVocab: Integer;                               // Active model vocabulary size; normally set from nSymbols.
  TokenID: TIVector;                             // Same as TokenizedCorpus.
  TokenizerKind: TTokenizerKind;                 // WesChat or GPT2Chat tokenizer;
  NewModel: Boolean = True;                      // If new model, initialize params.
  ParamsNeedCopyToDevice: Boolean = True;        // True  = host parameters need to be sent to GPU. False = current parameters are already on GPU
  CorpusPresent: Boolean = False;                // Parts of program present.
  SymbolTablePresent: Boolean = False;           // Do I need these?
  MergeTablePresent: Boolean = False;
  TokenizedCorpusPresent: Boolean = False;
  ModelPresent: Boolean = False;
  QueryPresent: Boolean = False;
const
  // Boundaries and starts.
  FirstTagToken = 260;
  UDTagBoundary = 400;

  // Basic byte/special tokens.
  TokBOS = 256;
  TokEOS = 257;
  TokPAD = 258;
  TokNUL = 259;

  // Parts of speech.
  TokNoun  = 260;   // |noun
  TokVerb  = 261;   // |verb
  TokAdj   = 262;   // |adj
  TokAdv   = 263;   // |adv
  TokPrep  = 264;   // |prep
  TokDet   = 265;   // |det
  TokPron  = 266;   // |pron
  TokAux   = 267;   // |aux
  TokSConj = 268;   // |sconj
  TokCConj = 269;   // |cconj
  TokPart  = 270;   // |part
  TokIntj  = 271;   // |intj
  TokNum   = 272;   // |num
  TokPropn = 273;   // |propn
  TokX     = 274;   // |x
  TokSym   = 275;   // |sym
  TokPunct = 276;   // |punct

  // Number.
  TokSing = 277;    // |sg
  TokPlur = 278;    // |pl

  // Person.
  TokPerson1 = 279; // |1p
  TokPerson2 = 280; // |2p
  TokPerson3 = 281; // |3p

  // Case.
  TokNom = 282;     // |nom
  TokAcc = 283;     // |acc
  TokGen = 284;     // |gen
  TokDat = 285;     // |dat
  TokLoc = 286;     // |loc
  TokIns = 287;     // |ins
  TokVoc = 288;     // |voc

  // Gender.
  TokMasc   = 289;  // |masc
  TokFem    = 290;  // |fem
  TokNeut   = 291;  // |neut
  TokCommon = 292;  // |common

  // Tense.
  TokPast = 293;    // |past
  TokPres = 294;    // |pres
  TokFut  = 295;    // |fut

  // Mood.
  TokMoodInd  = 296; // |ind
  TokMoodImp  = 297; // |imp
  TokMoodSub  = 298; // |sub
  TokMoodCond = 299; // |cond
  TokMoodOpt  = 300; // |opt

  // Verb form.
  TokVerbFin        = 301; // |fin
  TokVerbInf        = 302; // |inf
  TokVerbGer        = 303; // |ger
  TokVerbPart       = 304; // |participle
  TokVerbConv       = 305; // |conv

  // Voice.
  TokVoiceAct  = 306; // |act
  TokVoicePass = 307; // |pass
  TokVoiceMid  = 308; // |mid

  // Aspect.
  TokAspectImp   = 309; // |impf
  TokAspectPerf  = 310; // |perf
  TokAspectProg  = 311; // |prog
  TokAspectProsp = 312; // |prosp

  // Degree.
  TokDegreePos = 313; // |pos
  TokDegreeCmp = 314; // |cmp
  TokDegreeSup = 315; // |sup
  TokDegreeAbs = 316; // |abs

  // Definiteness.
  TokDefiniteDef = 317; // |def
  TokDefiniteInd = 318; // |indef

  // Pronoun type.
  TokPronArt = 319; // |art
  TokPronDem = 320; // |dem
  TokPronInt = 321; // |int
  TokPronPrs = 322; // |prs
  TokPronRel = 323; // |rel
  TokPronInd = 324; // |indpron
  TokPronNeg = 325; // |negpron
  TokPronTot = 326; // |tot

  // Possessive.
  TokPoss = 327;    // |poss

  // Reflexive.
  TokRefl = 328;    // |refl

  // Polarity.
  TokPolarityNeg = 329; // |neg
  TokPolarityPos = 330; // |positive

  // Numeral type.
  TokNumCard = 331; // |card
  TokNumOrd  = 332; // |ord
  TokNumFrac = 333; // |frac
  TokNumMult = 334; // |mult
  TokNumSets = 335; // |sets
  TokNumDist = 336; // |dist

  // Numeral form.
  TokNumDigit = 337; // |digit
  TokNumWord  = 338; // |numword
  TokNumRoman = 339; // |roman

  // Miscellaneous lexical properties.
  TokAbbr    = 340; // |abbr
  TokForeign = 341; // |foreign
  TokTypo    = 342; // |typo

  // Animacy.
  TokAnim     = 343; // |anim
  TokInan     = 344; // |inan
  TokHuman    = 345; // |human
  TokNonHuman = 346; // |nonhuman

  // First ID available for ordinary learned symbols is FirstMergedToken - 400.

implementation

begin
end.
1111
