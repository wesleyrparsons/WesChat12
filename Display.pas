unit Display;

{$mode ObjFPC}{$H+}{$I proprietary.txt}

{ WesChat, Version 1.2, begun January 10, 2026, by Wesley R. Parsons, wespar@bellouth.net, www.wesparsons.com.}

interface

uses
  Global;

// Pause procedures.
procedure HardPauseNNL;
procedure HardPause;
procedure Pause;
procedure PauseNNL;

// Interrupt procedures.
function CheckForControlKey: Char;

// Display symbols procedures.
function CleanUpSymbol(const x: RawByteString): RawByteString;
procedure DisplayByteSymbolTable(const SymbolTable: TSymbolTable);
function ConsoleText(const S: UnicodeString): UnicodeString;

// Display vectors and matrices.
procedure DisplayVector(const V: TIVector);
procedure DisplayX(const X: TSeqMatrix; const Part: TPart = B); overload;
procedure VTPDisplayX(const Mess: string; const X: TSeqMatrix; const Part: TPart = B); overload;
procedure DisplayX(const X: THiddenMatrix; const Part: TPart = B); overload;
procedure VTPDisplayX(const Mess: string; const X: THiddenMatrix; const Part: TPart = B); overload;
procedure DisplayX(const X: TSeqVocabMatrix; const Part: TPart = B); overload;
procedure VTPDisplayX(const Mess: string; const X: TSeqVocabMatrix; const Part: TPart = B); overload;
procedure DisplayX(const X: TEmbeddingsMatrix; const Part: TPart = B); overload;
procedure VTPDisplayX(const Mess: string; const X: TEmbeddingsMatrix; const Part: TPart = B); overload;
procedure DisplayX(const X: TScoresMatrix; const Part: TPart = B); overload;
procedure VTPDisplayX(const Mess: string; const X: TScoresMatrix; const Part: TPart = B); overload;

// Report information on program.
procedure ReportKeyVariables;
procedure ReportProgramInfo;

implementation

uses
  Crt,
  DateUtils,
  Math,
  SysUtils;

var
  EmbeddingParams, AttentionParams, FFNParams, LayerNormParams, BlockParams, TotalParams: Int64;

// Pause, unconditional, no new line.
procedure HardPauseNNL;
begin
  Write('Hit <CR> to continue.... ');
  Readln;
end;

// Pause, unconditional, new line.
procedure HardPause;
begin
  HardPauseNNL;
  Writeln;
end;

// Pause, subject to DoNotPause.
procedure Pause;
var
  tt: TDateTime;
begin
  if not DoNotPause then begin
    tt := Now;
    HardPause;
    Stoptime := StopTime + Now - tt;
  end;
end;

// Pause, subject to DoNotPause, no new line.
procedure PauseNNL;
var
  tt: TDateTime;
begin
  if not DoNotPause then begin
    tt := Now;
    HardPauseNNL;
    Stoptime := StopTime + Now - tt;
  end;
end;

// Returns key pressed.
function CheckForControlKey: Char;
begin
  if KeyPressed then
    Result := ReadKey
  else
    Result := #0;  // No key.
end;

// Compute trainable parameters.
procedure ComputeTrainableParameters;
begin
  EmbeddingParams := Int64(nVocab) * ModelDim;

  // Wq, Wk, Wv, W0.
  AttentionParams := Int64(4) * ModelDim * ModelDim;

  // W1, W2, b1, b2.
  FFNParams := Int64(2) * ModelDim * ModelDimProj + ModelDimProj + ModelDim;

  // Gamma1, Beta1, Gamma2, Beta2.
  LayerNormParams := Int64(4) * ModelDim;

  BlockParams := AttentionParams + FFNParams + LayerNormParams;
  TotalParams := EmbeddingParams + Int64(nBlock) * BlockParams;
end;

procedure FullReportTrainableParameters;
var
  AllBlockParams: Int64;
begin
  ComputeTrainableParameters;
  AllBlockParams := Int64(nBlock) * BlockParams;

  Writeln('--- Summary Specs and Trainable Parameters Calculation ---');
  Writeln('Trainable Parameters: Embeddings, Wq, Wk, Wv, W0, W1, b1, W2, b2, Gamma1, Beta1, Gamma2, Beta2');
  Writeln('nVocab        = ', Global.nVocab);
  Writeln('ModelDim      = ', ModelDim);
  Writeln('ModelDimProj  = ', ModelDimProj);
  Writeln('nBlock        = ', nBlock);

  Writeln('--- Detailed Trainable Parameter Calculation ---');
  Writeln('Embeddings:');
  Writeln('  nVocab * ModelDim');
  Writeln('  ', Global.nVocab, ' * ', ModelDim, ' = ', EmbeddingParams);
  Writeln('Attention parameters per block:');
  Writeln('  4 * ModelDim * ModelDim');
  Writeln('  4 * ', ModelDim, ' * ', ModelDim, ' = ', AttentionParams, ' (Includes Wq, Wk, Wv, and W0).');
  Writeln('FFN parameters per block:');
  Writeln('  2 * ModelDim * ModelDimProj + ModelDimProj + ModelDim');
  Writeln('  2 * ', ModelDim, ' * ', ModelDimProj, ' + ', ModelDimProj, ' + ', ModelDim, ' = ', FFNParams, '  Includes W1, W2, b1, and b2.');
  Writeln('LayerNorm parameters per block:');
  Writeln('  4 * ModelDim');
  Writeln('  4 * ', ModelDim, ' = ', LayerNormParams, '  Includes Gamma1, Beta1, Gamma2, and Beta2.');
  Writeln('Total parameters per transformer block:');
  Writeln('  Attention + FFN + LayerNorm');
  Writeln('  ', AttentionParams, ' + ', FFNParams, ' + ', LayerNormParams, ' = ', BlockParams);
  Writeln('All transformer blocks:');
  Writeln('  nBlock * BlockParams');
  Writeln('  ', nBlock, ' * ', BlockParams, ' = ', AllBlockParams);
  Writeln('Total trainable parameters:');
  Writeln('  Embeddings + all transformer blocks');
  Writeln('  ', EmbeddingParams, ' + ', AllBlockParams, ' = ', TotalParams);
end;

// Short report of trainable parameters.
function NumberTrainableParameters: Int64;
begin
  ComputeTrainableParameters;
  Result := TotalParams;
end;

// Write key variables in program.
procedure ReportKeyVariables;
begin
  Writeln('Maximum symbols: ', MaxSymbols, '; Maximum epochs (MaxEpochs): ', MaxEpochs, '.');
  Case LearningStyle of
    SlowLearning:
      Writeln('Learning rate (slow): 0..10: 0.01; 11..20: 0.005; 21..100: 0.0005; 101..300: 0.0001; else 0.00005.');
    FastLearning:
      Writeln('Learning rate (fast): 0..30: 0.01; 31..100: 0.005; 101..800: 0.001; else 0.0005.');
    RolledOffLearning:
      Writeln('Learning rate (rolled off): Floor LR = ', FloorLearningRate: 9: 7, ' Base LR = ', BaseLearningRate: 9: 7, ' LR rolloff = ', RollOff: 9: 7, '.');
  end;
  Writeln('Weight decay: ', WeightDecay: 9: 7, '; Clip limit: ', ClipLimit: 9: 7, '; Dropouts used: ', Training, '.');
  Writeln('Number of trainable parameters is ', NumberTrainableParameters, '.');
end;

// Report path.
procedure ReportPath(const PathLabel, PathValue: string);
begin
  if Trim(PathValue) = '' then
    Writeln(PathLabel, ': (none)')
  else
    Writeln(PathLabel, ': ', PathValue);
end;

// Write information on state of program.
procedure ReportProgramInfo;
begin
  Writeln('--- Program Information ---');
  Writeln('WesChat, Version: ', Version);
  Writeln('Author: Wesley R. Parsons');
  Writeln('Date: begun January 10, 2026');
  Writeln('--- Folder Paths ---');
  ReportPath('Existing work root', ExistingWorkRoot);
  ReportPath('Working directory', WorkingDir);
  ReportPath('Work root', WorkRoot);  Writeln('Sequence Length (SeqLen): ', SeqLen);
  Writeln('--- Model Dimensions ---');
  Writeln('Stride: (Stride) ', Stride);
  Writeln('Model Dimensions (ModelDim): ', ModelDim);
  Writeln('Dimensional Projections (Proj): ', Proj);
  Writeln('Heads (nHead): ', nHead);
  Writeln('Blocks (nBlock): ', nBlock);
  Writeln('Epochs (MaxEpochs): ', MaxEpochs);
  Writeln('Maximum Vocabulary (MaxVocab): ', DimVocab);
  Writeln('Number of Vocabulary (nVocab): ', Global.nVocab);
  Writeln('--- Model Specs ---');
  Case LearningStyle of
    SlowLearning:
      // Display slow learning rate schedule.
      Writeln('Learning rate (slow): ', LearningRate: 9: 7, ' with 0..10: 0.01; 11..20: 0.005; 21..100: 0.0005; 101..300: 0.0001; else 0.00005. ');
    FastLearning:
      // Display fast learning rate schedule.
      Writeln('Learning rate (fast): ', LearningRate: 9: 7, ' with 0..30: 0.01; 31..100: 0.005; 101..800: 0.001; else 0.0005. ');
    RolledOffLearning:
      // Learning Rolled off learning rate.
      Writeln('Learning rate (rolled of): ', LearningRate: 9: 7, ' Floor LR = ', FloorLearningRate: 9: 7, ' Base LR = ', BaseLearningRate: 9: 7, ' LR rolloff = ', RollOff: 9: 7, '.');
  end;
  if OverrideLearningRate <> -1.0 then
    Writeln('Override Learning Rate: ', OverrideLearningRate: 9 :7);
  Writeln('Current Learning Rate: ', LearningRate: 9: 7);
  Writeln('Weight decay: ', WeightDecay: 9: 7);
  Writeln('Clip limit: ', ClipLimit: 9: 7);
  Writeln('Temperature: ', TTemperature: 9: 7);
  Writeln('Global step: ', GlobalStep);
  Writeln('Dropouts for Attention, MLP, Residual (A, MLP, RDropout): ', ADropout: 4: 4, ' ', MLPDropout: 4: 4, ' ', RDropout: 4: 4);
  FullReportTrainableParameters;
end;

// Replace unprintable symbols with space.
function CleanUpSymbol(const x: RawByteString): RawByteString;
var
  j, L: Integer;
  ch: Char;
begin
  L := Length(x);
  SetLength(Result, L);   // Allocate output string.

  for j := 1 to L do begin
    ch := x[j];

    if Ord(ch) in [1..31, 127..255] then
      Result[j] := ' '
    else
      Result[j] := ch;
  end;
end;

// Display the symbol table.
procedure DisplayByteSymbolTable(const SymbolTable: TSymbolTable);
var
  i: Integer;
begin
  Writeln('--- Symbol Table ---');
  for i := 0 to High(SymbolTable) do begin  // Loop thru each symbol in table.
    if (i in [0..31]) or (i in [127..255]) then
      Write(i: 8, IntToHex(i, 2): 15)       // Hex for non-ASCII characters.
    else
      Write(i: 8, '"' + SymbolTable[i] + '"': 15);
    if (i mod 5) = 4 then Writeln;
    if (i > 0) and (i mod 100 = 99) then Pause;
  end;
  Writeln('Symbol table length = ', Length(SymbolTable));
end;

// Display for inference CR/LF.
function ConsoleText(const S: UnicodeString): UnicodeString;
begin
  Result := StringReplace(S, #13#10, #10, [rfReplaceAll]);
  Result := StringReplace(Result, #13, #10, [rfReplaceAll]);
  Result := StringReplace(Result, #10, #13#10, [rfReplaceAll]);
end;

// Display a vector, character by character, then pause.
procedure DisplayVector(const V: TIVector);
var
  i: Integer;
begin
  for i := 0 to High(V) do   // V is an array of integers.
    Write(V[i], ' ');
  Writeln;
  Pause;
end;

// Display scope of display below.
procedure PartScope(const Part: TPart);
begin
  Case Part of
    B: Writeln(' Beginning.');
    E: Writeln(' End.');
    F: Writeln(' Full.');
    G: Writeln(' Sample.');
  end;
end;

// Display an X matrix, B, E, F, or G.
procedure DisplayX(const X: TSeqMatrix; const Part: TPart = B); overload;
const
  tStride = 10;
var
  i, j, iB, iE, jB, jE: Integer;
  vStride: Integer = 1;
  hStride: Integer = 1;
begin
  Case Part of
    B: begin
      iB := 0;
      iE := 9;
      jB := 0;
      jE := 9;
    end;
    E: begin
      iB := High(X) - 9;
      iE := High(X);
      jB := High(X[0]) - 9;
      jE := High(X[0]);
    end;
    F: begin
      iB := 0;
      iE := High(X);
      jB := 0;
      jE := High(X[0]);
    end;
    G: begin
      vStride := Max(1, Floor(Length(X) / tStride));
      hStride := Max(1, Floor(Length(X[0]) / tStride));
      iB := 0;
      iE := tStride - 2;
      jB := 0;
      jE := tStride - 2;
    end;
  end;
  Write('       ');
  for j := jB to jE do
    Write(j * hStride: 8, '    ');
  if Part = G then
    Write(High(X[0]): 8, '    ');
  Writeln;
  for i := iB to iE do begin
    Write(i * vStride: 4);
    for j := jB to jE do
      Write(X[i * vStride, j * hStride]: 11: 5, ' ');
    if Part = G then
      Write(X[i * vStride, High(X[0])]: 11: 5, ' ');
    Writeln;
  end;
  if Part = G then begin
    Write(High(X): 4);
    for j := jB to jE do
      Write(X[High(X), j * hStride]: 11: 5, ' ');
    Write(X[High(X), High(X[0])]: 11: 5, ' ');
    Writeln;
  end;
end;

// Conditional form of DisplayX.
procedure VTPDisplayX(const Mess: string; const X: TSeqMatrix; const Part: TPart = B); overload;
begin
  if VerboseTransform then begin
    Write(Mess);
    PartScope(Part);
    DisplayX(X, Part);
    Pause;
  end;
end;

// Display a Hidden matrix, B, E, F, or G.
procedure DisplayX(const X: THiddenMatrix; const Part: TPart = B); overload;
const
  tStride = 10;
var
  i, j, iB, iE, jB, jE: Integer;
  vStride: Integer = 1;
  hStride: Integer = 1;
begin
  Case Part of
    B: begin
      iB := 0;
      iE := 9;
      jB := 0;
      jE := 9;
    end;
    E: begin
      iB := High(X) - 9;
      iE := High(X);
      jB := High(X[0]) - 9;
      jE := High(X[0]);
    end;
    F: begin
      iB := 0;
      iE := High(X);
      jB := 0;
      jE := High(X[0]);
    end;
    G: begin
      vStride := Max(1, Floor(Length(X) / tStride));
      hStride := Max(1, Floor(Length(X[0]) / tStride));
      iB := 0;
      iE := tStride - 2;
      jB := 0;
      jE := tStride - 2;
    end;
  end;
  Write('       ');
  for j := jB to jE do
    Write(j * hStride: 8, '    ');
  if Part = G then
    Write(High(X[0]): 8, '    ');
  Writeln;
  for i := iB to iE do begin
    Write(i * vStride: 4);
    for j := jB to jE do
      Write(X[i * vStride, j * hStride]: 11: 5, ' ');
    if Part = G then
      Write(X[i * vStride, High(X[0])]: 11: 5, ' ');
    Writeln;
  end;
  if Part = G then begin
    Write(High(X): 4);
    for j := jB to jE do
      Write(X[High(X), j * hStride]: 11: 5, ' ');
    Write(X[High(X), High(X[0])]: 11: 5, ' ');
    Writeln;
  end;
end;

// Conditional form of DisplayX.
procedure VTPDisplayX(const Mess: string; const X: THiddenMatrix; const Part: TPart = B); overload;
begin
  if VerboseTransform then begin
    Write(Mess);
    PartScope(Part);
    DisplayX(X, Part);
    Pause;
  end;
end;

// Display a Embeddings matrix, B, E, F, or G.
procedure DisplayX(const X: TEmbeddingsMatrix; const Part: TPart = B); overload;
const
  tStride = 10;
var
  i, j, iB, iE, jB, jE: Integer;
  vStride: Integer = 1;
  hStride: Integer = 1;
begin
  Case Part of
    B: begin
      iB := 0;
      iE := 9;
      jB := 0;
      jE := 9;
    end;
    E: begin
      iB := Max(0, nVocab - 10);
      iE := nVocab - 1;
      jB := High(X[0]) - 9;
      jE := High(X[0]);
    end;
    F: begin
      iB := 0;
      iE := nVocab - 1;
      jB := 0;
      jE := High(X[0]);
    end;
    G: begin
      vStride := Max(1, Floor(Length(X) / tStride));
      hStride := Max(1, Floor(Length(X[0]) / tStride));
      iB := 0;
      iE := tStride - 2;
      jB := 0;
      jE := tStride - 2;
    end;
  end;
  Write('       ');
  for j := jB to jE do
    Write(j * hStride: 8, '    ');
  if Part = G then
    Write(High(X[0]): 8, '    ');
  Writeln;
  for i := iB to iE do begin
    Write(i * vStride: 4);
    for j := jB to jE do
      Write(X[i * vStride, j * hStride]: 11: 5, ' ');
    if Part = G then
      Write(X[i * vStride, High(X[0])]: 11: 5, ' ');
    Writeln;
  end;
  if Part = G then begin
    Write(nVocab - 1: 4);
    for j := jB to jE do
      Write(X[nVocab - 1, j * hStride]: 11: 5, ' ');
    Write(X[nVocab - 1, High(X[0])]: 11: 5, ' ');
    Writeln;
  end;
end;

// Conditional form of DisplayX.
procedure VTPDisplayX(const Mess: string; const X: TEmbeddingsMatrix; const Part: TPart = B); overload;
begin
  if VerboseTransform then begin
    Write(Mess);
    PartScope(Part);
    DisplayX(X, Part);
    Pause;
  end;
end;

// Display a Vocab TSeq matrix, B, E, F, or G.
procedure DisplayX(const X: TSeqVocabMatrix; const Part: TPart = B); overload;
const
  tStride = 10;
var
  i, j, iB, iE, jB, jE: Integer;
  vStride: Integer = 1;
  hStride: Integer = 1;
begin
  Case Part of
    B: begin
      iB := 0;
      iE := 9;
      jB := 0;
      jE := 9;
    end;
    E: begin
      iB := High(X) - 9;
      iE := High(X);
      jB := High(X[0]) - 9;
      jE := High(X[0]);
    end;
    F: begin
      iB := 0;
      iE := High(X);
      jB := 0;
      jE := High(X[0]);
    end;
    G: begin
      vStride := Max(1, Floor(Length(X) / tStride));
      hStride := Max(1, Floor(Length(X[0]) / tStride));
      iB := 0;
      iE := tStride - 2;
      jB := 0;
      jE := tStride - 2;
    end;
  end;
  Write('       ');
  for j := jB to jE do
    Write(j * hStride: 8, '    ');
  if Part = G then
    Write(High(X[0]): 8, '    ');
  Writeln;
  for i := iB to iE do begin
    Write(i * vStride: 4);
    for j := jB to jE do
      Write(X[i * vStride, j * hStride]: 11: 7, ' ');
    if Part = G then
      Write(X[i * vStride, High(X[0])]: 11: 7, ' ');
    Writeln;
  end;
  if Part = G then begin
    Write(High(X): 4);
    for j := jB to jE do
      Write(X[High(X), j * hStride]: 11: 7, ' ');
    Write(X[High(X), High(X[0])]: 11: 7, ' ');
    Writeln;
  end;
end;

// Conditional form of DisplayX.
procedure VTPDisplayX(const Mess: string; const X: TSeqVocabMatrix; const Part: TPart = B); overload;
begin
  if VerboseTransform then begin
    Write(Mess);
    PartScope(Part);
    DisplayX(X, Part);
    Pause;
  end;
end;

// Display a ScoresHead matrix, B, E, F, or G.
procedure DisplayX(const X: TScoresMatrix; const Part: TPart = B); overload;
const
  tStride = 10;
var
  i, j, iB, iE, jB, jE: Integer;
  vStride: Integer = 1;
  hStride: Integer = 1;
begin
  Case Part of
    B: begin
      iB := 0;
      iE := 9;
      jB := 0;
      jE := 9;
    end;
    E: begin
      iB := High(X) - 9;
      iE := High(X);
      jB := High(X[0]) - 9;
      jE := High(X[0]);
    end;
    F: begin
      iB := 0;
      iE := High(X);
      jB := 0;
      jE := High(X[0]);
    end;
    G: begin
      vStride := Max(1, Floor(Length(X) / tStride));
      hStride := Max(1, Floor(Length(X[0]) / tStride));
      iB := 0;
      iE := tStride - 2;
      jB := 0;
      jE := tStride - 2;
    end;
  end;
  Write('       ');
  for j := jB to jE do
    Write(j * hStride: 8, '    ');
  if Part = G then
    Write(High(X[0]): 8, '    ');
  Writeln;
  for i := iB to iE do begin
    Write(i * vStride: 4);
    for j := jB to jE do
      Write(X[i * vStride, j * hStride]: 11: 7, ' ');
    if Part = G then
      Write(X[i * vStride, High(X[0])]: 11: 7, ' ');
    Writeln;
  end;
  if Part = G then begin
    Write(High(X): 4);
    for j := jB to jE do
      Write(X[High(X), j * hStride]: 11: 7, ' ');
    Write(X[High(X), High(X[0])]: 11: 7, ' ');
    Writeln;
  end;
end;

// Conditional form of DisplayX.
procedure VTPDisplayX(const Mess: string; const X: TScoresMatrix; const Part: TPart = B); overload;
begin
  if VerboseTransform then begin
    Write(Mess);
    PartScope(Part);
    DisplayX(X, Part);
    Pause;
  end;
end;

end.
