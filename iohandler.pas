unit IOHandler;

{$mode ObjFPC}{$H+}{$I proprietary.txt}

{ WesChat, Version 1.2, begun January 10, 2026, by Wesley R. Parsons, wespar@bellouth.net, www.wesparsons.com.}

interface

uses
  Classes,
  Crt,
  DateUtils,
  Display,
  FileUtil,
  Global,
  Util;

procedure ReadFileBytes(const FileName: string; var OneCorpus: TBVector);
procedure LoadSymbolTable(const FileName: string; var SymbolTable: TSymbolTable);
procedure LoadTokenList(const TokenFileName: string; var TokenizedCorpus: TIVector);
procedure SaveSymbolTable(const SymbolFileName: string; const SymbolTable: TSymbolTable);
procedure SaveTokenList(const TokenizedCorpus: TIVector; const TokenFileName: String);
procedure RestoreTrainingCheckpoint(const C: TTrainingCheckpoint);
function SaveModel(const FileName: string; var Model: TWModelParams): Boolean;
function LoadModel(const FileName: string; var Model: TWModelParams): Boolean;

implementation

// Read file of raw bytes, one by one.
procedure ReadFileBytes(const FileName: String; var OneCorpus: TBVector);
var
  F: File;
  Size, i: Integer;
  B: Byte;
begin
  AssignFile(F, FileName);
  Reset(F, 1);     // Open in binary mode.
  Size := FileSize(F);
  SetLength(OneCorpus, Size);

  // Write the Corpus as it is read.
  if VeryVerboseTokenize then
    Writeln('--- Original Corpus ---');
  for i := 0 to Size - 1 do begin
    BlockRead(F, B, 1);
    OneCorpus[i] := B;

    if VeryVerboseTokenize then
      if DisplayEachByteRead then
        if B < 32 then
          Write('<', B, '>')
        else
          Write(Chr(B));
  end;
  CloseFile(F);

  if VeryVerboseTokenize then begin
    Writeln('ReadByteFile: ');
    for i := 0 to 150 do
      Write(OneCorpus[i], ' ');
    Readln;
  end;
  if VeryVerboseTokenize then
    Writeln;

  // Display initial Corpus length.
  Writeln('Read ', Size, ' bytes from ', FileName, '.');
end;

// Load the symbol table from file. IOHandler.
procedure LoadSymbolTable(const FileName: string; var SymbolTable: TSymbolTable);
var
  F: file;
  Magic: array[0..3] of Char;
  S: string;
  i, Len: Integer;
begin
  BOS := 256;
  EOS := 257;
  PAD := 258;
  UNK := 259;
  Assign(F, FileName);
  Reset(F, 1);

  // Magic header.
  BlockRead(F, Magic, SizeOf(Magic));
  if (Magic[0] <> 'S') or (Magic[1] <> 'Y') or
     (Magic[2] <> 'M') or (Magic[3] <> 'T') then begin
    Close(F);
    Writeln('Invalid symbol table file.');
    Pause;
    Exit;
  end;

  // New version, 16 bytes.
  BlockRead(F, Version, 16);
  // Old Churchill, 4 bytes.
  // BlockRead(F, OldVersion, SizeOf(OldVersion));

  // Symbol count.
  BlockRead(F, nSymbols, SizeOf(nSymbols));
  SetLength(SymbolTable, NSymbols);

  // Meta symbol IDs.
  BlockRead(F, BOS, SizeOf(BOS));
  BlockRead(F, EOS, SizeOf(EOS));
  BlockRead(F, PAD, SizeOf(PAD));
  BlockRead(F, UNK, SizeOf(UNK));

  // Read symbols.
  for i := 0 to nSymbols - 1 do begin
    BlockRead(F, Len, SizeOf(Len));
    SetLength(S, Len);
    if Len > 0 then
      BlockRead(F, S[1], Len);
    SymbolTable[i] := S;
  end;

  Close(F);
  nSymbols := Length(SymbolTable);
  Writeln('Loaded ', nSymbols, ' symbols from ', FileName, '.');
end;

// Save symbol table.
procedure SaveSymbolTable(const SymbolFileName: string; const SymbolTable: TSymbolTable);
var
  F: file;
  NumSymbols: Integer;
  i, Len: Integer;
begin
  Assign(F, SymbolFileName);
  ReWrite(F, 1);

  // Magic.
  BlockWrite(F, SymbolMagic, SizeOf(SymbolMagic));

  // Version.
  BlockWrite(F, Version, 16);

  // Symbol count.
  NumSymbols := Length(SymbolTable);
  BlockWrite(F, NumSymbols, SizeOf(NumSymbols));

  // Special token IDs.
  BlockWrite(F, BOS, SizeOf(BOS));
  BlockWrite(F, EOS, SizeOf(EOS));
  BlockWrite(F, PAD, SizeOf(PAD));
  BlockWrite(F, UNK, SizeOf(UNK));

  // Write each symbol.
  for i := 0 to NumSymbols - 1 do begin
    Len := Length(SymbolTable[i]);
    BlockWrite(F, Len, SizeOf(Len));
    if Len > 0 then
      BlockWrite(F, SymbolTable[i][1], Len);
  end;

  Close(F);
  Writeln('File ', SymbolFileName, ' successfully saved.');
end;

// Load tokenized corpus from a token file.
procedure LoadTokenList(const TokenFileName: string; var TokenizedCorpus: TIVector);
var
  F: file of Integer;
  v, i, Count: Integer;
begin
  AssignFile(F, TokenFileName);
  Reset(F);

  // Determine number of tokens in file.
  Count := FileSize(F);

  // Allocate TokenizedCorpus.
  SetLength(TokenizedCorpus, Count);

  // Read them back.
  for i := 0 to Count - 1 do begin
    Read(F, v);
    TokenizedCorpus[i] := v;
  end;

  CloseFile(F);
  nTokenizedCorpus := Length(TokenizedCorpus);
  Writeln('Loaded ', Count, ' tokens from ', TokenFileName);
end;

// Save the output tokenized corpus to a token file.
procedure SaveTokenList(const TokenizedCorpus: TIVector; const TokenFileName: String);
var
  F: file of Integer;
  v, i: Integer;
begin
  AssignFile(F, TokenFileName);
  ReWrite(F);

  for i := 0 to High(TokenizedCorpus) do begin
    v := TokenizedCorpus[i];
    Write(F, v);
  end;

  CloseFile(F);
  Writeln('File ', TokenFileName, ' successfully saved.');
end;

// Clear pointers read from Model.ParamBlock.
procedure ClearDevicePointers(var Model: TWModelParams);
var
  b: Integer;
begin
  Model.Embeddings.dValue := nil;
  Model.Embeddings.dGrad := nil;

  for b := 0 to nBlock - 1 do begin
    with Model.ParamBlock[b] do begin
      Wq.dValue := nil; Wq.dGrad := nil;
      Wk.dValue := nil; Wk.dGrad := nil;
      Wv.dValue := nil; Wv.dGrad := nil;
      W0.dValue := nil; W0.dGrad := nil;
      W1.dValue := nil; W1.dGrad := nil;
      W2.dValue := nil; W2.dGrad := nil;
      b1.dValue := nil; b1.dGrad := nil;
      b2.dValue := nil; b2.dGrad := nil;
      Gamma1.dValue := nil; Gamma1.dGrad := nil;
      Beta1.dValue := nil; Beta1.dGrad := nil;
      Gamma2.dValue := nil; Gamma2.dGrad := nil;
      Beta2.dValue := nil; Beta2.dGrad := nil;
    end;
  end;
end;

// Capture and restore variables to save.
procedure CaptureTrainingCheckpoint(out C: TTrainingCheckpoint);
begin
  C.GlobalStep := GlobalStep;
  C.CompletedEpochs := CompletedEpochs;

  C.LearningStyle := LearningStyle;
  C.LearningRate := LearningRate;
  C.OverrideLearningRate := OverrideLearningRate;
  C.BaseLearningRate := BaseLearningRate;
  C.FloorLearningRate := FloorLearningRate;
  C.RollOff := RollOff;

  C.WeightDecay := WeightDecay;
  C.ClipLimit := ClipLimit;
  C.TTemperature := TTemperature;
  C.ITemperature := ITemperature;

  C.ADropOut := ADropOut;
  C.RDropOut := RDropOut;
  C.MLPDropOut := MLPDropOut;

  if ShuffleWindows then
    C.ShuffleWindows := True
  else
    C.ShuffleWindows := False;

  C.Stride := Stride;
  C.StartStride := StartStride;
  C.GlobalSeed := GlobalSeed;
end;

procedure RestoreTrainingCheckpoint(const C: TTrainingCheckpoint);
begin
  GlobalStep := C.GlobalStep;
  CompletedEpochs := C.CompletedEpochs;
  LearningStyle := C.LearningStyle;
  LearningRate := C.LearningRate;
  OverrideLearningRate := C.OverrideLearningRate;
  BaseLearningRate := C.BaseLearningRate;
  FloorLearningRate := C.FloorLearningRate;
  RollOff := C.RollOff;
  WeightDecay := C.WeightDecay;
  ClipLimit := C.ClipLimit;
  TTemperature := C.TTemperature;
  ITemperature := C.ITemperature;
  ADropOut := C.ADropOut;
  RDropOut := C.RDropOut;
  MLPDropOut := C.MLPDropOut;
  ShuffleWindows := C.ShuffleWindows;
  Stride := C.Stride;
  StartStride := C.StartStride;
  GlobalSeed := C.GlobalSeed;
  DecayScale := 1.0 - LearningRate * WeightDecay;       // Derived value.
end;

// Save a model.
function SaveModel(const FileName: string; var Model: TWModelParams): Boolean;
var
  F: file;
  IOModelDim, IONVocab, IONBlock, IOSeqLen,
    IODimVocab, IOModelDimProj, IOProj, IONHead: Integer;
  Checkpoint: TTrainingCheckpoint;
begin
  Result := False;

  if CudaAllocated then
    CopyParamsToHost(Model);

  IOModelDim     := ModelDim;
  IOModelDimProj := ModelDimProj;
  IOProj         := Proj;
  IONVocab       := nVocab;
  IODimVocab     := DimVocab;
  IONBlock       := nBlock;
  IONHead        := nHead;
  IOSeqLen       := SeqLen;

  CaptureTrainingCheckpoint(Checkpoint);

  AssignFile(F, FileName);
  try
    Rewrite(F, 1);

    BlockWrite(F, SymbolMagic, SizeOf(SymbolMagic));
    BlockWrite(F, Version, SizeOf(Version));
    BlockWrite(F, IOModelDim,     SizeOf(IOModelDim));
    BlockWrite(F, IOModelDimProj, SizeOf(IOModelDimProj));
    BlockWrite(F, IOProj,         SizeOf(IOProj));
    BlockWrite(F, IONVocab,       SizeOf(IONVocab));
    BlockWrite(F, IODimVocab,     SizeOf(IODimVocab));
    BlockWrite(F, IONBlock,       SizeOf(IONBlock));
    BlockWrite(F, IONHead,        SizeOf(IONHead));
    BlockWrite(F, IOSeqLen,       SizeOf(IOSeqLen));

    BlockWrite(F, Model, SizeOf(Model));
    BlockWrite(F, Checkpoint, SizeOf(Checkpoint));
    CloseFile(F);
    Result := True;
  except
    try
      CloseFile(F);
    except
    end;
    Result := False;
  end;
end;

// Load a model.
function LoadModel(const FileName: string; var Model: TWModelParams): Boolean;
var
  F: file;
var
  FileMagic: array[0..3] of Char;
  IOModelDim, IONVocab, IONBlock, IOSeqLen,
    IODimVocab, IOModelDimProj, IOProj, IONHead: Integer;
  Checkpoint: TTrainingCheckpoint;
  HasTrainingCheckpoint: Boolean;
begin
  Result := False;

  AssignFile(F, FileName);
  try
    Reset(F, 1);

    BlockRead(F, FileMagic, SizeOf(FileMagic));
    HasTrainingCheckpoint := False;

    if (FileMagic[0] = 'W') and (FileMagic[1] = 'E') and (FileMagic[2] = 'S') and (FileMagic[3] = '2') then
      HasTrainingCheckpoint := True
    else if not ((FileMagic[0] = 'W') and (FileMagic[1] = 'E') and (FileMagic[2] = 'S') and (FileMagic[3] = '1')) then begin
      CloseFile(F);
      Writeln('Invalid model file.');
      Exit;
    end;

    BlockRead(F, Version, SizeOf(Version));
    BlockRead(F, IOModelDim,     SizeOf(IOModelDim));
    BlockRead(F, IOModelDimProj, SizeOf(IOModelDimProj));
    BlockRead(F, IOProj,         SizeOf(IOProj));
    BlockRead(F, IONVocab,       SizeOf(IONVocab));
    BlockRead(F, IODimVocab,     SizeOf(IODimVocab));
    BlockRead(F, IONBlock,       SizeOf(IONBlock));
    BlockRead(F, IONHead,        SizeOf(IONHead));
    BlockRead(F, IOSeqLen,       SizeOf(IOSeqLen));

    if IOModelDim <> ModelDim then begin
      CloseFile(F);
      Writeln('ModelDim mismatch. File = ', IOModelDim, ' Program = ', ModelDim);
      Exit;
    end;

    if IONBlock <> nBlock then begin
      CloseFile(F);
      Writeln('nBlock mismatch. File = ', IONBlock, ' Program = ', nBlock);
      Exit;
    end;

    if IOModelDimProj <> ModelDimProj then begin
      Writeln('ModelDimProj mismatch. File = ', IOModelDimProj, ' Program = ', ModelDimProj);
      CloseFile(F);
      Exit;
    end;

    if IOProj <> Proj then begin
      Writeln('Proj mismatch. File = ', IOProj, ' Program = ', Proj);
      CloseFile(F);
      Exit;
    end;

    if IODimVocab <> DimVocab then begin
      Writeln('DimVocab mismatch. File = ', IODimVocab, ' Program = ', DimVocab);
      CloseFile(F);
      Exit;
    end;

    if IONVocab > DimVocab then begin
      Writeln('nVocab in file exceeds DimVocab. File = ', IONVocab, ' Program DimVocab = ', DimVocab);
      CloseFile(F);
      Exit;
    end;

    if IONHead <> nHead then begin
      Writeln('nHead mismatch. File = ', IONHead, ' Program = ', nHead);
      CloseFile(F);
      Exit;
    end;

    if IOSeqLen <> SeqLen then begin
      Writeln('SeqLen mismatch. File = ', IOSeqLen, ' Program = ', SeqLen);
      CloseFile(F);
      Exit;
    end;

    BlockRead(F, Model, SizeOf(Model));

    if HasTrainingCheckpoint then begin
      BlockRead(F, Checkpoint, SizeOf(Checkpoint));
      RestoreTrainingCheckpoint(Checkpoint)
    end
    else begin
      // Old WES1 model: training state was not stored.
      GlobalStep := 0;
      CompletedEpochs := 0;

      Writeln('Old WES1 model loaded.');
      Writeln('Training settings were not stored in this model.');
      Writeln('Current program training settings will be used.');
    end;

    CloseFile(F);

    nVocab := IONVocab;
    ClearDevicePointers(Model);
    NewModel := False;
    Result := True;
  except
    try
      CloseFile(F);
    except
    end;
    Result := False;
  end;
end;
end.
