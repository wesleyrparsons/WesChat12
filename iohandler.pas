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
  Math,
  SysUtils,
  Util;

// Read, load, and save data.
procedure ReadFileBytes(const FileName: string; var OneCorpus: TBVector);
procedure LoadSymbolTable(const FileName: string; var SymbolTable: TSymbolTable);
procedure LoadTokenList(const TokenFileName: string; var TokenizedCorpus: TIVector);
procedure SaveSymbolTable(const SymbolFileName: string; const SymbolTable: TSymbolTable);
procedure SaveSymbolTableIfMissing(const BaseName: string);
procedure SaveTokenList(const TokenizedCorpus: TIVector; const TokenFileName: String);
procedure RestoreTrainingCheckpoint(const C: TTrainingCheckpoint);
function SaveModel(const FileName: string; var Model: TWModelParams; var AdamWState: TWAdamWState): Boolean;
function LoadModel(const FileName: string; var Model: TWModelParams; var AdamWState: TWAdamWState): Boolean;

// Convert Tiny stories text files.
procedure ConvertTSEndOfText(const FileName: string);
procedure ConvertTSSeparators(const FileName: string);

implementation
const
  SymbolMagic3: array[0..7] of Char = ('W','E','S','3','S','Y','M','T');
  TokenMagic3: array[0..7] of Char = ('W','E','S','3','T','O','K','L');
  ModelMagic3: array[0..7] of Char = ('W','E','S','3','M','O','D','L');
  WES3ModelHeaderSize = 1024;
  WES3FileInfoSize = 1024;
  WES3ModelHeaderVersion = 1;
  WES3CheckpointVersion = 1;
  WES3OptimizerVersion = 1;

type
  TSymbolFileFormat = (sffWES3, sffWES2, sffOld);
  TTokenFileFormat = (tffWES3, tffWES2, tffOld);
  TWES3FileInfo = packed record
    HeaderVersion: UInt32;
    HeaderSize: UInt32;
    TokenizerKind: UInt32;
    Flags: UInt32;
    CorpusBytes: Int64;
    ProcessedBytes: Int64;
    RawTokenCount: Int64;
    TokenCount: Int64;
    PaddingCount: Int64;
    SymbolCount: UInt32;
    MergeCount: UInt32;
    CorpusID: UInt64;
    ProgramVersion: array[0..15] of Char;
    Reserved: array[0..935] of Byte;
  end;
  {$if SizeOf(TWES3FileInfo) <> WES3FileInfoSize}
    {$fatal TWES3FileInfo must be exactly 1024 bytes}
  {$endif}
  TWES3ModelHeader = packed record
    HeaderVersion: UInt32;
    HeaderSize: UInt32;
    CheckpointVersion: UInt32;
    OptimizerVersion: UInt32;
    TokenizerKind: UInt32;
    Flags: UInt32;
    CorpusID: QWord;
    CorpusBytes: Int64;
    RawTokenCount: Int64;
    TokenCount: Int64;
    PaddingCount: Int64;
    SymbolCount: UInt32;
    MergeCount: UInt32;
    ModelDim: UInt32;
    ModelDimProj: UInt32;
    Proj: UInt32;
    NVocab: UInt32;
    DimVocab: UInt32;
    NBlock: UInt32;
    NHead: UInt32;
    SeqLen: UInt32;
    ProgramVersion: array[0..15] of Char;
    Reserved: array[0..903] of Byte;
  end;
  {$if SizeOf(TWES3ModelHeader) <> WES3ModelHeaderSize}
    {$fatal TWES3ModelHeader must be exactly 1024 bytes}
  {$endif}

// Compute a corpus ID to distinguish and check corpora.
  function ComputeCorpusID(const OneCorpus: TBVector): QWord;
  const
    FNVOffsetBasis: QWord = 14695981039346656037;
    FNVPrime: QWord = 1099511628211;
  var
    i: Integer;
  begin
    Result := FNVOffsetBasis;

    for i := 0 to High(OneCorpus) do begin
      Result := Result xor QWord(OneCorpus[i]);
      Result := Result * FNVPrime;
    end;
  end;

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
  //CorpusID := ComputeCorpusID(OneCorpus);
end;

// Load the symbol table from file.
procedure LoadSymbolTable(const FileName: string; var SymbolTable: TSymbolTable);
var
  F: file;
  FileMagic: array[0..7] of Char;
  FileInfo: TWES3FileInfo;
  S: string;
  i, Len: Integer;
  SymbolFileFormat: TSymbolFileFormat;
begin
  BOS := 256;
  EOS := 257;
  PAD := 258;
  UNK := 259;

  Assign(F, FileName);
  Reset(F, 1);

  // Read first 8 bytes so we can identify WES3, WES2, or old SYMT.
  BlockRead(F, FileMagic, SizeOf(FileMagic));

  // WES3SYMT.
  if
    (FileMagic[0] = 'W') and (FileMagic[1] = 'E') and
    (FileMagic[2] = 'S') and (FileMagic[3] = '3') and
    (FileMagic[4] = 'S') and (FileMagic[5] = 'Y') and
    (FileMagic[6] = 'M') and (FileMagic[7] = 'T') then begin
    SymbolFileFormat := sffWES3;
  end

  // WES2SYMT.
  else if
    (FileMagic[0] = 'W') and (FileMagic[1] = 'E') and
    (FileMagic[2] = 'S') and (FileMagic[3] = '2') and
    (FileMagic[4] = 'S') and (FileMagic[5] = 'Y') and
    (FileMagic[6] = 'M') and (FileMagic[7] = 'T') then begin
    SymbolFileFormat := sffWES2;
  end

  // Old 4-byte SYMT format.
  else if
    (FileMagic[0] = 'S') and (FileMagic[1] = 'Y') and
    (FileMagic[2] = 'M') and (FileMagic[3] = 'T') then begin
    SymbolFileFormat := sffOld;
    Seek(F, 4);
  end

  else begin
    Close(F);
    Writeln('Invalid symbol table file.');
    Pause;
    Exit;
  end;

  case SymbolFileFormat of

    sffWES3: begin
      // File pointer is already immediately after WES3SYMT.
      BlockRead(F, FileInfo, SizeOf(FileInfo));

      if FileInfo.HeaderSize <> WES3ModelHeaderSize then begin
        Close(F);
        Writeln('Invalid WES3 symbol table header size: ', FileInfo.HeaderSize, '.');
        Pause;
        Exit;
      end;

      // Restore metadata.
      nCorpus := FileInfo.CorpusBytes;
      RawTokenCount := FileInfo.RawTokenCount;
      nSymbols := FileInfo.SymbolCount;

      // Restore program version if desired.
      Move(FileInfo.ProgramVersion[0], Version[0], SizeOf(FileInfo.ProgramVersion));

      SetLength(SymbolTable, nSymbols);
    end;

    sffWES2, sffOld: begin
      // WES2 is already positioned after its 8-byte magic.
      // Old SYMT was repositioned to byte 4 above.

      // Version, 16 bytes.
      BlockRead(F, Version, 16);

      // Symbol count.
      BlockRead(F, nSymbols, SizeOf(nSymbols));
      SetLength(SymbolTable, nSymbols);
    end;

  end;

  // Special token IDs. Same position relative to symbol data in all formats.
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

  case SymbolFileFormat of
    sffWES3: Writeln('Loaded WES3 symbol table with ', nSymbols, ' symbols from ', FileName, '.');
    sffWES2: Writeln('Loaded WES2 symbol table with ', nSymbols, ' symbols from ', FileName, '.');
    sffOld:  Writeln('Loaded old symbol table with ', nSymbols, ' symbols from ', FileName, '.');
  end;
end;
{// Load the symbol table from file.
procedure LoadSymbolTable(const FileName: string; var SymbolTable: TSymbolTable);
var
  F: file;
  FileMagic: array[0..7] of Char;
  S: string;
  i, Len: Integer;
  NewFormat: Boolean;
begin
  BOS := 256;
  EOS := 257;
  PAD := 258;
  UNK := 259;

  Assign(F, FileName);
  Reset(F, 1);

  // Read first 8 bytes so we can identify either format.
  BlockRead(F, FileMagic, SizeOf(FileMagic));

  // New WES2 symbol-table format.
  NewFormat :=
    (FileMagic[0] = 'W') and (FileMagic[1] = 'E') and
    (FileMagic[2] = 'S') and (FileMagic[3] = '2') and
    (FileMagic[4] = 'S') and (FileMagic[5] = 'Y') and
    (FileMagic[6] = 'M') and (FileMagic[7] = 'T');

  if NewFormat then begin
    // Already positioned immediately after 8-byte magic.
  end

  // Old 4-byte SYMT format.
  else if
    (FileMagic[0] = 'S') and (FileMagic[1] = 'Y') and
    (FileMagic[2] = 'M') and (FileMagic[3] = 'T') then begin
    Seek(F, 4);
  end

  else begin
    Close(F);
    Writeln('Invalid symbol table file.');
    Pause;
    Exit;
  end;

  // Version, 16 bytes.
  BlockRead(F, Version, 16);

  // Symbol count.
  BlockRead(F, nSymbols, SizeOf(nSymbols));
  SetLength(SymbolTable, nSymbols);

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
end;}

// Save WES3 symbol table.
procedure SaveSymbolTable(const SymbolFileName: string; const SymbolTable: TSymbolTable);
var
  F: file;
  FileInfo: TWES3FileInfo;
  NumSymbols: Integer;
  i, Len: Integer;
begin
  FillChar(FileInfo, SizeOf(FileInfo), 0);

  NumSymbols := Length(SymbolTable);

  FileInfo.HeaderVersion := 1;
  FileInfo.HeaderSize := WES3ModelHeaderSize;

  FileInfo.TokenizerKind := Ord(TokenizerKind);
  FileInfo.Flags := 0;

  FileInfo.CorpusBytes := nCorpus;
  FileInfo.ProcessedBytes := nCorpus;

  FileInfo.RawTokenCount := RawTokenCount;
  FileInfo.TokenCount := nTokenizedCorpus;
  FileInfo.PaddingCount := nTokenizedCorpus - RawTokenCount;

  FileInfo.SymbolCount := NumSymbols;
  FileInfo.MergeCount := 0;
  FileInfo.CorpusID := 0;

  Move(Version[0], FileInfo.ProgramVersion[0], Min(SizeOf(Version), SizeOf(FileInfo.ProgramVersion)));

  Assign(F, SymbolFileName);
  Rewrite(F, 1);

  // WES3SYMT magic.
  BlockWrite(F, SymbolMagic3, SizeOf(SymbolMagic3));

  // Fixed 1024-byte WES3 metadata header.
  BlockWrite(F, FileInfo, SizeOf(FileInfo));

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

{procedure SaveSymbolTable(const SymbolFileName: string; const SymbolTable: TSymbolTable);
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
end;}

// Save symbol table if not in \symbols.
procedure SaveSymbolTableIfMissing(const BaseName: string);
var
  FileName: string;
begin
  FileName := SymbolDir + ChangeFileExt(CleanBaseName(BaseName), '.sym');

  if FileExists(FileName) then begin
    Writeln('Symbol table already exists: ', FileName);
    Exit;
  end;

  SaveSymbolTable(FileName, SymbolTable);
  Writeln('Symbol table saved: ', FileName);
end;

// Load tokenized corpus from a token file.
procedure LoadTokenList(const TokenFileName: string; var TokenizedCorpus: TIVector);
var
  F: file;
  FileMagic: array[0..7] of Char;
  FileInfo: TWES3FileInfo;
  TokenFileFormat: TTokenFileFormat;
  v, i: Integer;
  Count: Int64;
begin
  AssignFile(F, TokenFileName);
  Reset(F, 1);

  if FileSize(F) < SizeOf(Integer) then begin
    CloseFile(F);
    Writeln('Invalid or empty token file.');
    Pause;
    Exit;
  end;

  // Read first 8 bytes if available so we can identify WES3 or WES2.
  if FileSize(F) >= SizeOf(FileMagic) then begin
    BlockRead(F, FileMagic, SizeOf(FileMagic));

    // WES3TOKL.
    if
      (FileMagic[0] = 'W') and (FileMagic[1] = 'E') and
      (FileMagic[2] = 'S') and (FileMagic[3] = '3') and
      (FileMagic[4] = 'T') and (FileMagic[5] = 'O') and
      (FileMagic[6] = 'K') and (FileMagic[7] = 'L') then begin
      TokenFileFormat := tffWES3;
    end

    // WES2TOKL.
    else if
      (FileMagic[0] = 'W') and (FileMagic[1] = 'E') and
      (FileMagic[2] = 'S') and (FileMagic[3] = '2') and
      (FileMagic[4] = 'T') and (FileMagic[5] = 'O') and
      (FileMagic[6] = 'K') and (FileMagic[7] = 'L') then begin
      TokenFileFormat := tffWES2;
    end

    // Old token file with no header.
    else begin
      TokenFileFormat := tffOld;
      Seek(F, 0);
    end;
  end
  else begin
    TokenFileFormat := tffOld;
    Seek(F, 0);
  end;

  case TokenFileFormat of

    tffWES3: begin
      // File pointer is already immediately after WES3TOKL.
      BlockRead(F, FileInfo, SizeOf(FileInfo));

      if FileInfo.HeaderSize <> WES3ModelHeaderSize then begin
        CloseFile(F);
        Writeln('Invalid WES3 token-file header size: ', FileInfo.HeaderSize, '.');
        Pause;
        Exit;
      end;

      Count := FileInfo.TokenCount;

      if Count < 0 then begin
        CloseFile(F);
        Writeln('Invalid token count in WES3 token file.');
        Pause;
        Exit;
      end;

      // Restore available WES3 metadata.
      nCorpus := FileInfo.CorpusBytes;
      RawTokenCount := FileInfo.RawTokenCount;
      nTokenizedCorpus := FileInfo.TokenCount;

      // Add these as you implement/use the corresponding globals.
      // ProcessedCorpusBytes := FileInfo.ProcessedBytes;
      // PaddingCount := FileInfo.PaddingCount;
      // TokenizerKind := TTokenizerKind(FileInfo.TokenizerKind);
      // nSymbols := FileInfo.SymbolCount;
      // nMerges := FileInfo.MergeCount;
      // CorpusID := FileInfo.CorpusID;
    end;

    tffWES2: begin
      // File pointer is already immediately after WES2TOKL.
      Count := (FileSize(F) - SizeOf(FileMagic)) div SizeOf(Integer);
    end;

    tffOld: begin
      // Raw list of Integer tokens.
      Count := FileSize(F) div SizeOf(Integer);
    end;

  end;

  SetLength(TokenizedCorpus, Count);

  for i := 0 to Count - 1 do begin
    BlockRead(F, v, SizeOf(v));
    TokenizedCorpus[i] := v;
  end;

  CloseFile(F);

  nTokenizedCorpus := Length(TokenizedCorpus);

  // WES3 already stores the true unpadded token count.
  // Older formats must reconstruct it by removing trailing PAD tokens.
  if TokenFileFormat <> tffWES3 then begin
    RawTokenCount := Length(TokenizedCorpus);

    while (RawTokenCount > 0) and
      (TokenizedCorpus[RawTokenCount - 1] = PAD) do
      Dec(RawTokenCount);
  end;

  case TokenFileFormat of
    tffWES3: Writeln('Loaded WES3 token list with ', Count, ' tokens from ', TokenFileName, '.');
    tffWES2: Writeln('Loaded WES2 token list with ', Count, ' tokens from ', TokenFileName, '.');
    tffOld:  Writeln('Loaded old token list with ', Count, ' tokens from ', TokenFileName, '.');
  end;
end;
{// Load tokenized corpus from a token file.
procedure LoadTokenList(const TokenFileName: string; var TokenizedCorpus: TIVector);
var
  F: file;
  FileMagic: array[0..7] of Char;
  v, i: Integer;
  Count: Int64;
  HasMagic: Boolean;
begin
  AssignFile(F, TokenFileName);
  Reset(F, 1);

  HasMagic := False;

  // Check for the 8-byte WES2TOKL header.
  if FileSize(F) >= SizeOf(FileMagic) then begin
    BlockRead(F, FileMagic, SizeOf(FileMagic));

    HasMagic :=
      (FileMagic[0] = 'W') and
      (FileMagic[1] = 'E') and
      (FileMagic[2] = 'S') and
      (FileMagic[3] = '2') and
      (FileMagic[4] = 'T') and
      (FileMagic[5] = 'O') and
      (FileMagic[6] = 'K') and
      (FileMagic[7] = 'L');
  end;

  if HasMagic then begin
    // File position is already immediately after the 8-byte header.
    Count := (FileSize(F) - SizeOf(FileMagic)) div SizeOf(Integer);
  end
  else begin
    // Old token file with no header. Return to beginning.
    Seek(F, 0);
    Count := FileSize(F) div SizeOf(Integer);
  end;

  SetLength(TokenizedCorpus, Count);

  for i := 0 to Count - 1 do begin
    BlockRead(F, v, SizeOf(v));
    TokenizedCorpus[i] := v;
  end;

  CloseFile(F);

  nTokenizedCorpus := Length(TokenizedCorpus);
  RawTokenCount := Length(TokenizedCorpus);

  while (RawTokenCount > 0) and
    (TokenizedCorpus[RawTokenCount - 1] = PAD) do
    Dec(RawTokenCount);

  Writeln('Loaded ', Count, ' tokens from ', TokenFileName, '.');
end;}

// Save WES3 token list.
procedure SaveTokenList(const TokenizedCorpus: TIVector; const TokenFileName: String);
var
  F: file;
  FileInfo: TWES3FileInfo;
  TokenCount, PaddingCount: Int64;
  i, v: Integer;
begin
  FillChar(FileInfo, SizeOf(FileInfo), 0);

  TokenCount := Length(TokenizedCorpus);
  PaddingCount := TokenCount - RawTokenCount;

  // WES3 metadata.
  FileInfo.HeaderVersion := 1;
  FileInfo.HeaderSize := WES3ModelHeaderSize;

  FileInfo.TokenizerKind := Ord(TokenizerKind);
  FileInfo.Flags := 0;

  FileInfo.CorpusBytes := nCorpus;
  FileInfo.ProcessedBytes := nCorpus;

  FileInfo.RawTokenCount := RawTokenCount;
  FileInfo.TokenCount := TokenCount;
  FileInfo.PaddingCount := PaddingCount;

  FileInfo.SymbolCount := nSymbols;
  FileInfo.MergeCount := MergeCount;

  FileInfo.CorpusID := CorpusID;

  Move(Version[0], FileInfo.ProgramVersion[0], SizeOf(FileInfo.ProgramVersion));

  Assign(F, TokenFileName);
  Rewrite(F, 1);

  // WES3TOKL magic.
  BlockWrite(F, TokenMagic3, SizeOf(TokenMagic3));

  // Fixed 1024-byte WES3 metadata header.
  BlockWrite(F, FileInfo, SizeOf(FileInfo));

  // Token data.
  for i := 0 to High(TokenizedCorpus) do begin
    v := TokenizedCorpus[i];
    BlockWrite(F, v, SizeOf(v));
  end;

  Close(F);

  Writeln('Saved ', TokenCount, ' tokens to ', TokenFileName, '.');
end;
{// Save the output tokenized corpus to a token file.
procedure SaveTokenList(const TokenizedCorpus: TIVector; const TokenFileName: string);
var
  F: file;
  v, i: Integer;
begin
  AssignFile(F, TokenFileName);
  Rewrite(F, 1);

  BlockWrite(F, TokenMagic, SizeOf(TokenMagic));

  for i := 0 to High(TokenizedCorpus) do begin
    v := TokenizedCorpus[i];
    BlockWrite(F, v, SizeOf(v));
  end;

  CloseFile(F);
  Writeln('File ', TokenFileName, ' successfully saved.');
end;}

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

  C.ShuffleWindows := ShuffleWindows;

  C.Stride := Stride;
  C.StartStride := StartStride;
  C.GlobalSeed := GlobalSeed;

  C.AdamWStep := AdamWStep;
  C.AdamBeta1 := AdamBeta1;
  C.AdamBeta2 := AdamBeta2;
  C.AdamEpsilon := AdamEpsilon;

  // Adaptive learning-rate state.
  C.AdaptiveLR := AdaptiveLR;
  C.AdaptiveLRState := AdaptiveLRState;

  // Historical loss/best-model state.
  C.MinLoss := MinLoss;
  C.MinLossEpoch := MinLossEpoch;
  C.BestSavedLoss := BestSavedLoss;
  C.LastBestSaveEpoch := LastBestSaveEpoch;
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

  AdamWStep := C.AdamWStep;
  AdamBeta1 := C.AdamBeta1;
  AdamBeta2 := C.AdamBeta2;
  AdamEpsilon := C.AdamEpsilon;

  // Adaptive learning-rate state.
  AdaptiveLR := C.AdaptiveLR;
  AdaptiveLRState := C.AdaptiveLRState;

  // Historical loss/best-model state.
  MinLoss := C.MinLoss;
  MinLossEpoch := C.MinLossEpoch;
  BestSavedLoss := C.BestSavedLoss;
  LastBestSaveEpoch := C.LastBestSaveEpoch;

  // Derived value.
  DecayScale := 1.0 - LearningRate * WeightDecay;
end;
{procedure CaptureTrainingCheckpoint(out C: TTrainingCheckpoint);
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
  C.AdamWStep := AdamWStep;
  C.AdamBeta1 := AdamBeta1;
  C.AdamBeta2 := AdamBeta2;
  C.AdamEpsilon := AdamEpsilon;
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
  AdamWStep := C.AdamWStep;
  AdamBeta1 := C.AdamBeta1;
  AdamBeta2 := C.AdamBeta2;
  AdamEpsilon := C.AdamEpsilon;
  DecayScale := 1.0 - LearningRate * WeightDecay;       // Derived value.
end;}

// Save AdamW first and second moments.
// Only host M and V arrays are written; CUDA pointers are not written.
procedure SaveAdamWState(var F: file; const WAdamWState: TWAdamWState);
var
  k: Integer;
begin
  // Tied embeddings.
  with WAdamWState.Embeddings do begin
    BlockWrite(F, M, EmbeddingsSize);
    BlockWrite(F, V, EmbeddingsSize);
  end;

  // Per-block AdamW state.
  for k := 0 to nBlock - 1 do
    with WAdamWState.ParamBlock[k] do begin

      // Attention weights.
      BlockWrite(F, Wq.M, WeightSize);
      BlockWrite(F, Wq.V, WeightSize);

      BlockWrite(F, Wk.M, WeightSize);
      BlockWrite(F, Wk.V, WeightSize);

      BlockWrite(F, Wv.M, WeightSize);
      BlockWrite(F, Wv.V, WeightSize);

      BlockWrite(F, W0.M, WeightSize);
      BlockWrite(F, W0.V, WeightSize);

      // MLP weights.
      BlockWrite(F, W1.M, WeightProjectedSize);
      BlockWrite(F, W1.V, WeightProjectedSize);

      BlockWrite(F, W2.M, WeightProjectedSize);
      BlockWrite(F, W2.V, WeightProjectedSize);

      // Biases.
      BlockWrite(F, b1.M, ProjectedSize);
      BlockWrite(F, b1.V, ProjectedSize);

      BlockWrite(F, b2.M, ModelSize);
      BlockWrite(F, b2.V, ModelSize);

      // LayerNorm 1.
      BlockWrite(F, Gamma1.M, ModelSize);
      BlockWrite(F, Gamma1.V, ModelSize);

      BlockWrite(F, Beta1.M, ModelSize);
      BlockWrite(F, Beta1.V, ModelSize);

      // LayerNorm 2.
      BlockWrite(F, Gamma2.M, ModelSize);
      BlockWrite(F, Gamma2.V, ModelSize);

      BlockWrite(F, Beta2.M, ModelSize);
      BlockWrite(F, Beta2.V, ModelSize);
    end;
end;

// Save a WES3 model.
function SaveModel(const FileName: string; var Model: TWModelParams; var AdamWState: TWAdamWState): Boolean;
var
  F: file;
  Header: TWES3ModelHeader;
  Checkpoint: TTrainingCheckpoint;
  TokenCount, PaddingCount: Int64;
begin
  Result := False;

  if CudaAllocated then begin
    CopyParamsToHost(Model);
    CopyAdamWStateToHost(AdamWState);
  end;

  FillChar(Header, SizeOf(Header), 0);

  TokenCount := nTokenizedCorpus;

  if TokenCount >= RawTokenCount then
    PaddingCount := TokenCount - RawTokenCount
  else
    PaddingCount := 0;

  // WES3 file information.
  Header.HeaderVersion := WES3ModelHeaderVersion;
  Header.HeaderSize := WES3ModelHeaderSize;
  Header.CheckpointVersion := WES3CheckpointVersion;
  Header.OptimizerVersion := WES3OptimizerVersion;
  Header.Flags := 0;

  // Tokenizer/corpus information.
  Header.TokenizerKind := Ord(TokenizerKind);
  Header.CorpusID := CorpusID;
  Header.CorpusBytes := nCorpus;
  Header.RawTokenCount := RawTokenCount;
  Header.TokenCount := TokenCount;
  Header.PaddingCount := PaddingCount;
  Header.SymbolCount := nSymbols;

  // Use this if nMerges exists. Otherwise leave it zero.
  Header.MergeCount := nMerges;

  // Model architecture.
  Header.ModelDim := ModelDim;
  Header.ModelDimProj := ModelDimProj;
  Header.Proj := Proj;
  Header.NVocab := nVocab;
  Header.DimVocab := DimVocab;
  Header.NBlock := nBlock;
  Header.NHead := nHead;
  Header.SeqLen := SeqLen;

  // Program version.
  Move(Version[0], Header.ProgramVersion[0],
    Min(SizeOf(Version), SizeOf(Header.ProgramVersion)));

  // Capture everything needed to resume training.
  CaptureTrainingCheckpoint(Checkpoint);

  AssignFile(F, FileName);

  try
    Rewrite(F, 1);

    // WES3 model identification.
    BlockWrite(F, ModelMagic3, SizeOf(ModelMagic3));

    // Fixed 1024-byte WES3 header.
    BlockWrite(F, Header, SizeOf(Header));

    // Model parameters.
    BlockWrite(F, Model, SizeOf(Model));

    // Training state.
    BlockWrite(F, Checkpoint, SizeOf(Checkpoint));

    // AdamW M/V state.
    SaveAdamWState(F, AdamWState);

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
{function SaveModel(const FileName: string; var Model: TWModelParams; var AdamWState: TWAdamWState): Boolean;
var
  F: file;
  IOModelDim, IONVocab, IONBlock, IOSeqLen,
    IODimVocab, IOModelDimProj, IOProj, IONHead: Integer;
  Checkpoint: TTrainingCheckpoint;
begin
  Result := False;

  if CudaAllocated then begin
    CopyParamsToHost(Model);
    CopyAdamWStateToHost(AdamWState);
  end;

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

    BlockWrite(F, ModelMagic, SizeOf(ModelMagic));
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
    SaveAdamWState(F, AdamWState);
    CloseFile(F);
    Result := True;
  except
    try
      CloseFile(F);
    except
    end;
    Result := False;
  end;
end;}

// Load AdamW first and second moments.
// CUDA dM and dV pointers are allocated separately by MAllocCublas.
procedure LoadAdamWState(var F: file; var WAdamWState: TWAdamWState);
var
  k: Integer;
begin
  // Tied embeddings.
  with WAdamWState.Embeddings do begin
    BlockRead(F, M, EmbeddingsSize);
    BlockRead(F, V, EmbeddingsSize);
  end;

  // Per-block AdamW state.
  for k := 0 to nBlock - 1 do
    with WAdamWState.ParamBlock[k] do begin

      // Attention weights.
      BlockRead(F, Wq.M, WeightSize);
      BlockRead(F, Wq.V, WeightSize);

      BlockRead(F, Wk.M, WeightSize);
      BlockRead(F, Wk.V, WeightSize);

      BlockRead(F, Wv.M, WeightSize);
      BlockRead(F, Wv.V, WeightSize);

      BlockRead(F, W0.M, WeightSize);
      BlockRead(F, W0.V, WeightSize);

      // MLP weights.
      BlockRead(F, W1.M, WeightProjectedSize);
      BlockRead(F, W1.V, WeightProjectedSize);

      BlockRead(F, W2.M, WeightProjectedSize);
      BlockRead(F, W2.V, WeightProjectedSize);

      // Biases.
      BlockRead(F, b1.M, ProjectedSize);
      BlockRead(F, b1.V, ProjectedSize);

      BlockRead(F, b2.M, ModelSize);
      BlockRead(F, b2.V, ModelSize);

      // LayerNorm 1.
      BlockRead(F, Gamma1.M, ModelSize);
      BlockRead(F, Gamma1.V, ModelSize);

      BlockRead(F, Beta1.M, ModelSize);
      BlockRead(F, Beta1.V, ModelSize);

      // LayerNorm 2.
      BlockRead(F, Gamma2.M, ModelSize);
      BlockRead(F, Gamma2.V, ModelSize);

      BlockRead(F, Beta2.M, ModelSize);
      BlockRead(F, Beta2.V, ModelSize);
    end;
end;

// Load a WES3 model.
function LoadModel(const FileName: string; var Model: TWModelParams; var AdamWState: TWAdamWState): Boolean;
var
  F: file;
  FileMagic: array[0..7] of Char;
  Header: TWES3ModelHeader;
  Checkpoint: TTrainingCheckpoint;
begin
  Result := False;

  AssignFile(F, FileName);

  try
    Reset(F, 1);

    if FileSize(F) < SizeOf(FileMagic) + SizeOf(Header) then begin
      CloseFile(F);
      Writeln('Invalid or incomplete WES3 model file.');
      Exit;
    end;

    // Read and validate WES3MODL.
    BlockRead(F, FileMagic, SizeOf(FileMagic));

    if not (
      (FileMagic[0] = 'W') and
      (FileMagic[1] = 'E') and
      (FileMagic[2] = 'S') and
      (FileMagic[3] = '3') and
      (FileMagic[4] = 'M') and
      (FileMagic[5] = 'O') and
      (FileMagic[6] = 'D') and
      (FileMagic[7] = 'L')) then begin
      CloseFile(F);
      Writeln('Invalid model file. WES3MODL header not found.');
      Exit;
    end;

    // Read fixed WES3 model header.
    BlockRead(F, Header, SizeOf(Header));

    if Header.HeaderSize <> WES3ModelHeaderSize then begin
      CloseFile(F);
      Writeln('Invalid WES3 model header size. File = ', Header.HeaderSize,
        '; expected = ', WES3ModelHeaderSize, '.');
      Exit;
    end;

    if Header.HeaderVersion <> WES3ModelHeaderVersion then begin
      CloseFile(F);
      Writeln('Unsupported WES3 model header version. File = ',
        Header.HeaderVersion, '; expected = ', WES3ModelHeaderVersion, '.');
      Exit;
    end;

    if Header.CheckpointVersion <> WES3CheckpointVersion then begin
      CloseFile(F);
      Writeln('Unsupported checkpoint version. File = ',
        Header.CheckpointVersion, '; expected = ', WES3CheckpointVersion, '.');
      Exit;
    end;

    if Header.OptimizerVersion <> WES3OptimizerVersion then begin
      CloseFile(F);
      Writeln('Unsupported optimizer version. File = ',
        Header.OptimizerVersion, '; expected = ', WES3OptimizerVersion, '.');
      Exit;
    end;

    // Validate architecture against current program settings.
    if Header.ModelDim <> ModelDim then begin
      CloseFile(F);
      Writeln('ModelDim mismatch. File = ', Header.ModelDim,
        '; Program = ', ModelDim, '.');
      Exit;
    end;

    if Header.ModelDimProj <> ModelDimProj then begin
      CloseFile(F);
      Writeln('ModelDimProj mismatch. File = ', Header.ModelDimProj,
        '; Program = ', ModelDimProj, '.');
      Exit;
    end;

    if Header.Proj <> Proj then begin
      CloseFile(F);
      Writeln('Proj mismatch. File = ', Header.Proj,
        '; Program = ', Proj, '.');
      Exit;
    end;

    if Header.NVocab > DimVocab then begin
      CloseFile(F);
      Writeln('nVocab in model exceeds DimVocab. File = ', Header.NVocab,
        '; Program DimVocab = ', DimVocab, '.');
      Exit;
    end;

    if Header.DimVocab <> DimVocab then begin
      CloseFile(F);
      Writeln('DimVocab mismatch. File = ', Header.DimVocab,
        '; Program = ', DimVocab, '.');
      Exit;
    end;

    if Header.NBlock <> nBlock then begin
      CloseFile(F);
      Writeln('nBlock mismatch. File = ', Header.NBlock,
        '; Program = ', nBlock, '.');
      Exit;
    end;

    if Header.NHead <> nHead then begin
      CloseFile(F);
      Writeln('nHead mismatch. File = ', Header.NHead,
        '; Program = ', nHead, '.');
      Exit;
    end;

    if Header.SeqLen <> SeqLen then begin
      CloseFile(F);
      Writeln('SeqLen mismatch. File = ', Header.SeqLen,
        '; Program = ', SeqLen, '.');
      Exit;
    end;

    // Restore model metadata.
    TokenizerKind := TTokenizerKind(Header.TokenizerKind);
    CorpusID := Header.CorpusID;
    nCorpus := Header.CorpusBytes;
    RawTokenCount := Header.RawTokenCount;
    nTokenizedCorpus := Header.TokenCount;
    nSymbols := Header.SymbolCount;

    // nVocab may be smaller than DimVocab and should come from the model.
    nVocab := Header.NVocab;

    // Read model parameters.
    BlockRead(F, Model, SizeOf(Model));

    // Read saved training state.
    BlockRead(F, Checkpoint, SizeOf(Checkpoint));

    // Read AdamW M/V state.
    LoadAdamWState(F, AdamWState);

    CloseFile(F);

    RestoreTrainingCheckpoint(Checkpoint);
    AdamWStateLoaded := True;

    // Host record contains pointer values written into the file.
    // These must never be treated as valid device pointers.
    ClearDevicePointers(Model);

    NewModel := False;
    Result := True;

    Writeln('Loaded WES3 model: ', FileName, '.');

  except
    try
      CloseFile(F);
    except
    end;

    Result := False;
  end;
end;

// Convert the EOF story separators in tiny stories to char 254.
procedure ConvertTSEndOfText(const FileName: string);
const
  Marker: AnsiString = '<|endoftext|>';
var
  F: file;
  Corpus: TBVector;
  Source, Dest, j, BytesRead, Count: Integer;
  Match: Boolean;
begin
  if not FileExists(FileName) then begin
    Writeln('File not found: ', FileName);
    Exit;
  end;

  // Read entire file.
  Assign(F, FileName);
  Reset(F, 1);

  SetLength(Corpus, FileSize(F));

  if Length(Corpus) > 0 then
    BlockRead(F, Corpus[0], Length(Corpus), BytesRead)
  else
    BytesRead := 0;

  Close(F);

  if BytesRead <> Length(Corpus) then begin
    Writeln('Error reading file: ', FileName);
    Exit;
  end;

  // Replace <|endoftext|> with byte 254.
  Source := 0;
  Dest := 0;
  Count := 0;

  while Source < Length(Corpus) do begin
    Match := Source + Length(Marker) <= Length(Corpus);

    if Match then
      for j := 1 to Length(Marker) do
        if Corpus[Source + j - 1] <> Ord(Marker[j]) then begin
          Match := False;
          Break;
        end;

    if Match then begin
      Corpus[Dest] := 254;
      Inc(Dest);
      Inc(Source, Length(Marker));
      Inc(Count);
    end
    else begin
      Corpus[Dest] := Corpus[Source];
      Inc(Dest);
      Inc(Source);
    end;
  end;

  SetLength(Corpus, Dest);

  // Rewrite converted file.
  Assign(F, FileName);
  Rewrite(F, 1);

  if Length(Corpus) > 0 then
    BlockWrite(F, Corpus[0], Length(Corpus));

  Close(F);

  Writeln('Converted ', Count, ' <|endoftext|> separators in ', FileName, '.');
  Writeln('New file size = ', Length(Corpus), ' bytes.');
end;

// Convert the 3-byte story separators in tiny stories to char 254.
procedure ConvertTSSeparators(const FileName: string);
var
  F: file;
  Corpus: TBVector;
  Source, Dest, BytesRead, Count: Integer;
begin
  if not FileExists(FileName) then begin
    Writeln('File not found: ', FileName);
    Exit;
  end;

  // Read entire file.
  Assign(F, FileName);
  Reset(F, 1);

  SetLength(Corpus, FileSize(F));

  if Length(Corpus) > 0 then
    BlockRead(F, Corpus[0], Length(Corpus), BytesRead)
  else
    BytesRead := 0;

  Close(F);

  if BytesRead <> Length(Corpus) then begin
    Writeln('Error reading file: ', FileName);
    Exit;
  end;

  // Replace UTF-8 black square E2 96 A0 with byte 254.
  Source := 0;
  Dest := 0;
  Count := 0;

  while Source < Length(Corpus) do begin
    if (Source + 2 < Length(Corpus)) and
       (Corpus[Source] = 226) and
       (Corpus[Source + 1] = 150) and
       (Corpus[Source + 2] = 160) then begin

      Corpus[Dest] := 254;
      Inc(Dest);
      Inc(Source, 3);
      Inc(Count);
    end
    else begin
      Corpus[Dest] := Corpus[Source];
      Inc(Dest);
      Inc(Source);
    end;
  end;

  SetLength(Corpus, Dest);

  // Rewrite file with converted corpus.
  Assign(F, FileName);
  Rewrite(F, 1);

  if Length(Corpus) > 0 then
    BlockWrite(F, Corpus[0], Length(Corpus));

  Close(F);

  Writeln('Converted ', Count, ' TinyStories separators in ', FileName, '.');
  Writeln('New file size = ', Length(Corpus), ' bytes.');
end;
{// Load a model.
function LoadModel(const FileName: string; var Model: TWModelParams; var AdamWState: TWAdamWState): Boolean;
var
  F: file;
  FileMagic: array[0..7] of Char;
  IOModelDim, IONVocab, IONBlock, IOSeqLen,
    IODimVocab, IOModelDimProj, IOProj, IONHead: Integer;
  Checkpoint: TTrainingCheckpoint;
  OldWES2: Boolean;
begin
  Result := False;
  OldWES2 := False;

  AssignFile(F, FileName);
  try
    Reset(F, 1);

    // Read enough bytes to identify either new WES2MODL or old WES2.
    BlockRead(F, FileMagic, SizeOf(FileMagic));

    // New 8-byte model magic: WES2MODL.
    if (FileMagic[0] = 'W') and (FileMagic[1] = 'E') and
       (FileMagic[2] = 'S') and (FileMagic[3] = '2') and
       (FileMagic[4] = 'M') and (FileMagic[5] = 'O') and
       (FileMagic[6] = 'D') and (FileMagic[7] = 'L') then begin

      // File is already positioned immediately after the 8-byte magic.
    end

    // Older WES2 model with four-byte magic.
    else if (FileMagic[0] = 'W') and (FileMagic[1] = 'E') and
            (FileMagic[2] = 'S') and (FileMagic[3] = '2') then begin
      OldWES2 := True;
      Seek(F, 4);
    end

    else if (FileMagic[0] = 'S') and (FileMagic[1] = 'Y') and
            (FileMagic[2] = 'M') and (FileMagic[3] = 'T') then begin
      OldWES2 := True;
      Seek(F, 4);
    end

    else begin
      CloseFile(F);
      Writeln('Invalid model file.');
      Exit;
    end;

    BlockRead(F, Version,        SizeOf(Version));
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
      CloseFile(F);
      Writeln('ModelDimProj mismatch. File = ', IOModelDimProj, ' Program = ', ModelDimProj);
      Exit;
    end;

    if IOProj <> Proj then begin
      CloseFile(F);
      Writeln('Proj mismatch. File = ', IOProj, ' Program = ', Proj);
      Exit;
    end;

    if IODimVocab <> DimVocab then begin
      CloseFile(F);
      Writeln('DimVocab mismatch. File = ', IODimVocab, ' Program = ', DimVocab);
      Exit;
    end;

    if IONVocab > DimVocab then begin
      CloseFile(F);
      Writeln('nVocab in file exceeds DimVocab. File = ', IONVocab,
        ' Program DimVocab = ', DimVocab);
      Exit;
    end;

    if IONHead <> nHead then begin
      CloseFile(F);
      Writeln('nHead mismatch. File = ', IONHead, ' Program = ', nHead);
      Exit;
    end;

    if IOSeqLen <> SeqLen then begin
      CloseFile(F);
      Writeln('SeqLen mismatch. File = ', IOSeqLen, ' Program = ', SeqLen);
      Exit;
    end;

    BlockRead(F, Model, SizeOf(Model));

    // All supported models contain a WES2 training checkpoint and AdamW state.
    BlockRead(F, Checkpoint, SizeOf(Checkpoint));
    LoadAdamWState(F, AdamWState);
    RestoreTrainingCheckpoint(Checkpoint);
    AdamWStateLoaded := True;

    CloseFile(F);

    ClearDevicePointers(Model);
    NewModel := False;
    Result := True;

    Write('Loaded ');
    if OldWES2 then
      Write('older WES2 ');
    Writeln('model: ', FileName, '.');

  except
    try
      CloseFile(F);
    except
    end;
    Result := False;
  end;
end;}

end.
