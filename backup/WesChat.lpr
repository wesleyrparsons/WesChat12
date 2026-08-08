program WesChat;

{$mode ObjFPC}{$H+}{$I proprietary.txt}

{ WesChat, Version 1.2.
{ Note: Edited 7/26/2026 6 pm -- working from WesChat12 on OneDrive }
{ Need to recompile CEGradientStrided; Note Tiny Stories change in Symbolize }
{        Input Train        Input Query        Output
 Raw                        QueryString
 Bytes   Corpus             QueryCorpus
 Token   TokenizedCorpus    QueryTokenized     QueryOutput }
 Folder layout: WorkRoot \corpus \lists \logs \merges \models \scratch \symbols \tokens }

uses
  Classes,
  CombineTables,
  Crt,
  GPT2Tokenize,
  Display,
  FileUtil,
  Global,
  Infer,
  IOHandler,
  Matrix,
  OutputHead,
  ShellAPI,
  Symbolize,
  SysUtils,
  Train,
  Util,
  WesTokenize,
  Windows;

var
  // Corpus vars.
  Corpus: TBVector;
  TokenizedCorpus: TIVector;

  // Model vars.
  WModelParams: TWModelParams;
  WModelState: TWModelState;

  // File names.
  CorpusFileName, SymbolFileName, TokenFileName,
    ModelFileName, ListFile: string;

  // Current base name for default output filenames.
  CurrentBaseName: string = 'weschat';
  MinSymbols: Integer = 50;
  MinTokens: Integer = 50;
  MinCorpus: Integer = 50;
  Ch: string;
  CombinedSymbolTable: TSymbolTable;

// Work folder helpers
procedure OpenWorkFolderInExplorer;
begin
  if Trim(WorkRoot) = '' then begin
    Writeln('WorkRoot is blank.');
    Exit;
  end;

  ShellExecute(0, 'open', PChar(WorkRoot), nil, nil, SW_SHOWNORMAL);
end;

procedure NormalizeExistingWorkRoot;
begin
  ExistingWorkRoot := Trim(ExistingWorkRoot);

  if ExistingWorkRoot = '' then
    ExistingWorkRoot := 'C:\wc\';

  ExistingWorkRoot := IncludeTrailingPathDelimiter(ExpandFileName(ExistingWorkRoot));
end;

function ResolveWorkFolder(const FolderName: string): string;
var
  S: string;
begin
  NormalizeExistingWorkRoot;

  S := Trim(FolderName);

  if S = '' then
    S := 'WesChatWork';

  if IsAbsolutePath(S) then
    Result := ExpandFileName(S)
  else
    Result := ExpandFileName(ExistingWorkRoot + S);
end;

function TimeStamp: string;
begin
  Result := FormatDateTime('yyyy-mm-dd_hhnnss', Now);
end;

// Use the existing work root set above.
procedure SelectExistingWork(const FolderName, BaseName: string);
begin
  WorkingDir := ResolveWorkFolder(FolderName);
  InitWorkFolders(WorkingDir);

  CurrentBaseName := BaseName;
  WorkingName := BaseName;

  Writeln('Using predefined work folder: ', WorkRoot);
end;

function PathHasDirectory(const S: string): Boolean;
begin
  Result := ExtractFilePath(S) <> '';
end;

function ResolveInputFile(const UserName, PreferredDir: string): string;
begin
  Result := Trim(UserName);

  if Result = '' then
    Exit;

  if FileExists(Result) then begin
    Result := ExpandFileName(Result);
    Exit;
  end;

  if not PathHasDirectory(Result) then begin
    if FileExists(PreferredDir + Result) then begin
      Result := PreferredDir + Result;
      Exit;
    end;

    if FileExists(WorkRoot + Result) then begin
      Result := WorkRoot + Result;
      Exit;
    end;
  end;
end;

function MakeOutputFileName(const UserName, DefaultDir, BaseName, Ext: string): string;
var
  S: string;
begin
  S := Trim(UserName);

  if S = '' then
    S := ChangeFileExt(BaseName, Ext);

  if ExtractFileExt(S) = '' then
    S := S + Ext;

  if PathHasDirectory(S) then
    Result := ExpandFileName(S)
  else
    Result := DefaultDir + S;
end;

function DefaultSymbolStatsFile(const BaseName: string): string;
begin
  Result := LogDir + CleanBaseName(BaseName) + '.sym.tok';
end;

{function DefaultMetaFile(const BaseName: string): string;
begin
  Result := LogDir + ChangeFileExt(CleanBaseName(BaseName), '.meta');
end;}

function DefaultSymbolFile(const BaseName: string): string;
begin
  Result := SymbolDir + ChangeFileExt(CleanBaseName(BaseName), '.sym');
end;

function DefaultMergeFile(const BaseName: string): string;
begin
  Result := MergeDir + ChangeFileExt(CleanBaseName(BaseName), '.mer');
end;

function DefaultTokenFile(const BaseName: string): string;
begin
  Result := TokenDir + ChangeFileExt(CleanBaseName(BaseName), '.tok');
end;


function DefaultModelFile(const BaseName: string): string;
begin
  Result := ModelDir + CleanBaseName(BaseName) + '_' + TimeStamp + '.model';
end;

function DefaultTokenLogFile(const BaseName: string): string;
begin
  Result := LogDir + ChangeFileExt(CleanBaseName(BaseName), '.tok.log');
end;

function DefaultLogFile(const BaseName: string): string;
begin
  Result := LogDir + CleanBaseName(BaseName) + '_' + TimeStamp + '.log';
end;

procedure SaveSymbolizationFilesDefault(const BaseName: string);
begin
  if Length(SymbolTable) = 0 then begin
    Writeln('No symbol table to save.');
    Exit;
  end;

  Writeln('--- Saving Symbolization Files ---');

  SymbolFileName := DefaultSymbolFile(BaseName);
  SaveSymbolTable(SymbolFileName, SymbolTable);

  if Length(Merges) > 0 then
    SaveMergeTable(Merges, DefaultMergeFile(BaseName))
  else
    Writeln('No merges to save.');

  SaveMetaData(DefaultSymbolStatsFile(BaseName));
end;

procedure SaveTokenizationFilesDefault(const BaseName: string);
begin
  if Length(TokenizedCorpus) = 0 then begin
    Writeln('No token list to save.');
    Exit;
  end;

  Writeln('--- Saving Tokenization Files ---');

  TokenFileName := DefaultTokenFile(BaseName);
  SaveTokenList(TokenizedCorpus, TokenFileName);

  SaveTokenizationLog(TokenizedCorpus, DefaultTokenLogFile(BaseName));
end;

procedure WriteInfoLog(const BaseName: string);
var
  SaveOut: Text;
  LogName: string;
begin
  if not SaveFiles then Exit;

  LogName := DefaultLogFile(BaseName);

  SaveOut := Output;

  Assign(Output, LogName);
  Rewrite(Output);

  ReportProgramInfo;

  Close(Output);
  Output := SaveOut;

  Writeln('Log written: ', LogName);
end;

// Set tokenizer as WesTokenize or GPT2TTokenize.
procedure SetTokenizerMode(const NewTokenizer: TTokenizer);
begin
  Tokenizer := NewTokenizer;

  case Tokenizer of
    WesTokenizer: begin
      BOS := 256;
      EOS := 257;
      PAD := 258;
      UNK := 259;
      nVocab := nSymbols;
    end;

    GPT2Tokenizer: begin
      BOS := GPT2BOS;
      EOS := GPT2EOS;
      PAD := GPT2PAD;
      UNK := GPT2UNK;
      nVocab := GPT2ModelVocabSize;
    end;
  end;
end;

// General helpers.
function AskYesNo(const Prompt: string; DefaultYes: Boolean = True): Boolean;
var
  S: string;
begin
  Write(Prompt);

  if DefaultYes then
    Write(' (Y/n) ')
  else
    Write(' (y/N) ');

  Readln(S);
  S := UpperCase(Trim(S));

  if S = '' then
    Result := DefaultYes
  else
    Result := S[1] = 'Y';
end;

function AskChoice(const Prompt, Choices: string): string;
begin
  Write(Prompt, ' [', Choices, ']: ');
  Readln(Result);
  Result := UpperCase(Trim(Result));
end;

function RequireExistingFile(var FileName: string; const PreferredDir: string): Boolean;
begin
  FileName := ResolveInputFile(FileName, PreferredDir);
  Result := FileExists(FileName);

  if not Result then
    Writeln('File not found: ', FileName, '.');
end;

function RequireMinFileSize(const FileName: string; MinSize: Integer): Boolean;
begin
  Result := FileSize(FileName) >= MinSize;

  if not Result then
    Writeln('File too small: ', FileName, '. Size = ', FileSize(FileName),
      ' minimum = ', MinSize, '.');
end;

function ReadCorpusFilePrompt(var OutCorpusFileName: string; var OutCorpus: TBVector): Boolean;
begin
  Result := False;

  Write('Enter corpus file name: ');
  Readln(OutCorpusFileName);

  if not RequireExistingFile(OutCorpusFileName, CorpusDir) then
    Exit;

  if not RequireMinFileSize(OutCorpusFileName, MinCorpus) then
    Exit;

  ReadFileBytes(OutCorpusFileName, OutCorpus);
  nCorpus := Length(OutCorpus);

  CurrentBaseName := CleanBaseName(OutCorpusFileName);

  SetLength(CorpusFileNames, 1);
  CorpusFileNames[0] :=
    OutCorpusFileName + '   ' + IntToStr(FileSize(OutCorpusFileName)) +
    ' bytes   ' + DateTimeToStr(FileDateToDateTime(FileAge(OutCorpusFileName)));

  WriteInfoLog(CurrentBaseName);

  Result := True;
end;

// Append one integer vector onto another.
procedure AppendTokens(var Dest: TIVector; const Src: TIVector);
var
  OldLen, i: Integer;
begin
  OldLen := Length(Dest);
  SetLength(Dest, OldLen + Length(Src));

  for i := 0 to High(Src) do
    Dest[OldLen + i] := Src[i];
end;

// Read a file of file names and concatenate the corpuses.
procedure ProcessFileList(var ListFileName: string; var OutCorpus: TBVector);
var
  F: TextFile;
  Line, FullName, ListBaseDir: string;
  OneCorpus: TBVector;
  Count: Integer;
begin
  MultipleFileName := EmptyStr;

  Write('Enter name of file list: ');
  Readln(ListFileName);

  if not RequireExistingFile(ListFileName, ListDir) then
    Exit;

  CurrentBaseName := CleanBaseName(ListFileName);
  ListBaseDir := ExtractFilePath(ListFileName);

  AssignFile(F, ListFileName);
  Reset(F);

  Count := 0;
  FromSymbolTable := False;
  SetLength(OutCorpus, 0);
  SetLength(CorpusFileNames, 0);

  while not EOF(F) do begin
    ReadLn(F, Line);
    Line := Trim(Line);

    if Line = '' then
      Continue;

    FullName := Line;

    if not FileExists(FullName) then
      FullName := ListBaseDir + Line;

    if not FileExists(FullName) then
      FullName := CorpusDir + Line;

    if not FileExists(FullName) then begin
      Writeln('  File not found: ', Line, '.');
      Continue;
    end;

    if FileSize(FullName) < MinCorpus then begin
      Writeln('  Corpus too small, skipping: ', FullName);
      Continue;
    end;

    ReadFileBytes(FullName, OneCorpus);

    SetLength(CorpusFileNames, Count + 1);
    CorpusFileNames[Count] :=
      FullName + '   ' + IntToStr(FileSize(FullName)) + ' bytes   ' +
      DateTimeToStr(FileDateToDateTime(FileAge(FullName)));

    OutCorpus := Concat(OutCorpus, OneCorpus);
    nCorpus := Length(OutCorpus);

    Writeln('  File processed: ', FullName,
      '; corpus bytes read: ', Length(OneCorpus), '.');
    Writeln('  Total bytes read: ', Length(OutCorpus), '.');

    Inc(Count);
  end;

  CloseFile(F);

  Writeln('Combined corpus length = ', Length(OutCorpus));
  nCorpus := Length(OutCorpus);

  WriteInfoLog(CurrentBaseName);
end;

// Load helpers.
function LoadSymbolTablePrompt: Boolean;
begin
  Result := False;

  Write('Input symbol table file name: ');
  Readln(SymbolFileName);

  if not RequireExistingFile(SymbolFileName, SymbolDir) then Exit;

  FromSymbolTable := True;
  LoadSymbolTable(SymbolFileName, SymbolTable);
  ResetWesTrie;

  if Length(SymbolTable) < MinSymbols then begin
    Writeln('Too few symbols found. Length(SymbolTable) = ', Length(SymbolTable));
    Exit;
  end;

  nSymbols := Length(SymbolTable);
  nVocab := nSymbols;

  Writeln('Using symbol table: ', SymbolFileName);

  Result := True;
end;

function LoadTokenListPrompt: Boolean;
begin
  Result := False;

  Write('Enter token list file name: ');
  Readln(TokenFileName);

  if not RequireExistingFile(TokenFileName, TokenDir) then
    Exit;

  IOHandler.LoadTokenList(TokenFileName, TokenizedCorpus);

  if Length(TokenizedCorpus) < MinTokens then begin
    Writeln('Token list too small. Length = ', Length(TokenizedCorpus));
    Exit;
  end;

  PadToSeqMultiple(TokenizedCorpus, SeqLen);
  nTokenizedCorpus := Length(TokenizedCorpus);

  CurrentBaseName := CleanBaseName(TokenFileName);

  Writeln('Using token list: ', TokenFileName);

  Result := True;
end;

function LoadModelPrompt: Boolean;
begin
  Result := False;

  Write('Enter model file name: ');
  Readln(ModelFileName);

  if not RequireExistingFile(ModelFileName, ModelDir) then Exit;

  if CudaAllocated or (CuHandle <> nil) then
    EndCuda(WModelParams, WModelState);
  {if CudaAllocated then
    MDeallocateCublas(WModelParams, WModelState);}

  if LoadModel(ModelFileName, WModelParams) then begin
    Write('Loading model: ', ModelFileName);
    Write('; Model size: ', FileSize(ModelFileName), ' bytes');
    Writeln('; Model date: ', DateTimeToStr(FileDateToDateTime(FileAge(ModelFileName))), '.');
    NewModel := False;
    ParamsNeedCopyToDevice := True;
    Result := True;
  end
  else Writeln('File not loaded.');
end;

// Save helpers.
procedure SaveCurrentTokenListDefault;
begin
  if Length(TokenizedCorpus) = 0 then
    Exit;

  TokenFileName := DefaultTokenFile(CurrentBaseName);
  SaveTokenList(TokenizedCorpus, TokenFileName);
end;

procedure MaybeSaveTokenList;
var
  S: string;
begin
  if Length(TokenizedCorpus) = 0 then Exit;

  if AskYesNo('Save token list?', True) then begin
    Write('Output token list file name, blank for ',
      ExtractFileName(DefaultTokenFile(CurrentBaseName)), ': ');
    Readln(S);

    TokenFileName := MakeOutputFileName(S, TokenDir, CurrentBaseName, '.tok');
    SaveTokenList(TokenizedCorpus, TokenFileName);
  end;
end;

procedure MaybeSaveModel;
var
  S: string;
begin
  if AskYesNo('Save model?', True) then begin
    Write('Output model file name, blank for ',
      ExtractFileName(DefaultModelFile(CurrentBaseName)), ': ');
    Readln(S);

    ModelFileName := MakeOutputFileName(S, ModelDir,
      CurrentBaseName + '_' + TimeStamp, '.model');

    if SaveModel(ModelFileName, WModelParams) then
      Writeln('File ', ModelFileName, ' successfully saved.')
    else
      Writeln('File not saved.');
  end;
end;

// Wrappers.
procedure RunSymbolizeNoAutoSave(const InCorpus: TBVector);
var
  OldSaveFiles: Boolean;
begin
  OldSaveFiles := SaveFiles;
  SaveFiles := False;
  try
    RunSymbolize(InCorpus);
  finally
    SaveFiles := OldSaveFiles;
  end;
end;

procedure RunWesTokenizeNoAutoSave(const InCorpus: TBVector; var OutTokens: TIVector);
var
  OldSaveFiles: Boolean;
  OldSaveTokenizationFiles: Boolean;
begin
  OldSaveFiles := SaveFiles;
  OldSaveTokenizationFiles := SaveTokenizationFiles;

  SaveFiles := False;
  SaveTokenizationFiles := False;

  try
    RunWesTokenize(InCorpus, OutTokens);
  finally
    SaveTokenizationFiles := OldSaveTokenizationFiles;
    SaveFiles := OldSaveFiles;
  end;
end;

procedure RunGPT2TokenizeNoAutoSave(const InFileName: string; var OutTokens: TIVector);
var
  OldSaveFiles: Boolean;
begin
  OldSaveFiles := SaveFiles;
  SaveFiles := False;

  try
    RunGPT2Tokenize(InFileName, OutTokens);
  finally
    SaveFiles := OldSaveFiles;
  end;
end;

// Main workflow: K = Tokenize.
procedure TokenizeWithWes;
var
  SourceChoice, SymbolChoice: string;
  RawTokenCount, PaddedTokenCount: Integer;
begin
  SetLength(TokenizedCorpus, 0);

  SourceChoice := AskChoice('Corpus source: F = one file, L = list of corpus file names', 'F/L');

  if SourceChoice = 'L' then begin
    ProcessFileList(ListFile, Corpus);

    if Length(Corpus) < MinCorpus then begin
      Writeln('Combined corpus too small. Aborting tokenization.');
      Exit;
    end;
  end
  else begin
    if not ReadCorpusFilePrompt(CorpusFileName, Corpus) then Exit;
  end;

  SymbolChoice := AskChoice('Symbol table: C = create from corpus, S = load existing symbol table', 'C/S');

  if SymbolChoice = 'S' then begin
    if not LoadSymbolTablePrompt then Exit;
  end
  else begin
    FromSymbolTable := False;
    nSymbols := 0;
    nVocab := 0;
    SetLength(SymbolTable, 0);

    RunSymbolizeNoAutoSave(Corpus);
    ResetWesTrie;

    nSymbols := Length(SymbolTable);
    nVocab := nSymbols;

    if nSymbols < MinSymbols then begin
      Writeln('Too few symbols found after symbolization. nSymbols = ', nSymbols);
      Exit;
    end;

    if AskYesNo('Save symbolization files?', True) then
      SaveSymbolizationFilesDefault(CurrentBaseName);
  end;

  SetTokenizerMode(WesTokenizer);
  Writeln('Tokenizing with WesChat...');
  RunWesTokenizeNoAutoSave(Corpus, TokenizedCorpus);

  RawTokenCount := Length(TokenizedCorpus);

  PadToSeqMultiple(TokenizedCorpus, SeqLen);

  PaddedTokenCount := Length(TokenizedCorpus);
  nTokenizedCorpus := PaddedTokenCount;

  Write('WesChat tokenization complete.');
  Writeln(' Raw tokens = ', RawTokenCount, '; Padded tokens = ', PaddedTokenCount, '; Padding added = ', PaddedTokenCount - RawTokenCount, '; Symbols = ', nSymbols, '.');

  if AskYesNo('Save tokenization files?', True) then
    SaveTokenizationFilesDefault(CurrentBaseName);
end;

procedure TokenizeGPTSingleFile;
var
  PaddedTokenCount, RawTokenCount: Integer;
begin
  SetLength(TokenizedCorpus, 0);

  if not ReadCorpusFilePrompt(CorpusFileName, Corpus) then Exit;

  SetTokenizerMode(GPT2Tokenizer);
  RunGPT2TokenizeNoAutoSave(CorpusFileName, TokenizedCorpus);

  RawTokenCount := Length(TokenizedCorpus);
  PadToSeqMultiple(TokenizedCorpus, SeqLen);
  PaddedTokenCount := Length(TokenizedCorpus);
  nTokenizedCorpus := Length(TokenizedCorpus);

  Write('GPT tokenization complete.');
  Writeln(' Raw tokens = ', RawTokenCount, '; Padded tokens = ', PaddedTokenCount, '; Padding added = ', PaddedTokenCount - RawTokenCount,
    '; Vocabulary = ', nVocab, '.');

  MaybeSaveTokenList;
end;

procedure TokenizeGPTFileList;
var
  F: TextFile;
  Line, FullName, ListBaseDir: string;
  OneTokens: TIVector;
  Count: Integer;
begin
  SetLength(TokenizedCorpus, 0);

  Write('Enter name of file list: ');
  Readln(ListFile);

  if not RequireExistingFile(ListFile, ListDir) then Exit;

  CurrentBaseName := CleanBaseName(ListFile);
  ListBaseDir := ExtractFilePath(ListFile);

  AssignFile(F, ListFile);
  Reset(F);

  Count := 0;
  SetLength(CorpusFileNames, 0);

  while not EOF(F) do begin
    Readln(F, Line);
    Line := Trim(Line);

    if Line = '' then Continue;

    FullName := Line;

    if not FileExists(FullName) then
      FullName := ListBaseDir + Line;

    if not FileExists(FullName) then
      FullName := CorpusDir + Line;

    if not FileExists(FullName) then begin
      Writeln('  File not found: ', Line, '.');
      Continue;
    end;

    if FileSize(FullName) < MinCorpus then begin
      Writeln('  Corpus too small, skipping: ', FullName);
      Continue;
    end;

    SetLength(OneTokens, 0);
    RunGPT2TokenizeNoAutoSave(FullName, OneTokens);
    AppendTokens(TokenizedCorpus, OneTokens);

    SetLength(CorpusFileNames, Count + 1);
    CorpusFileNames[Count] := FullName + '   ' + IntToStr(FileSize(FullName)) + ' bytes   ' +
      DateTimeToStr(FileDateToDateTime(FileAge(FullName)));

    Inc(Count);

    Writeln('  GPT-tokenized: ', FullName, '; tokens added= ', Length(OneTokens), '; total tokens = ', Length(TokenizedCorpus), '.');
  end;

  CloseFile(F);

  SetTokenizerMode(GPT2Tokenizer);
  PadToSeqMultiple(TokenizedCorpus, SeqLen);
  nTokenizedCorpus := Length(TokenizedCorpus);

  Writeln('GPT list tokenization complete. Tokens = ', Length(TokenizedCorpus), '.');

  MaybeSaveTokenList;
end;

// Do a tokenization procedure.
procedure DoTokenize;
var
  TokChoice, SourceChoice: string;
begin
  Writeln;
  Writeln('--- Tokenize ---');
  Writeln('Creates a token list from one corpus file or from a list of corpus files.');
  Writeln('WesTokenize can create a new symbol table or use an existing one.');
  Writeln('GPT2Tokenize uses the GPT-2 vocabulary.');
  Writeln;

  TokChoice := AskChoice('Tokenizer: W = WesTokenize, G = GPT2Tokenize', 'W/G');

  if TokChoice = 'G' then begin
    EnsureGPT2VocabLoaded;

    SourceChoice := AskChoice('Corpus source: F = one file, L = list of corpus file names', 'F/L');

    if SourceChoice = 'L' then
      TokenizeGPTFileList
    else
      TokenizeGPTSingleFile;
  end
  else
    TokenizeWithWes;

  if Length(TokenizedCorpus) > 0 then begin
    if AskYesNo('Proceed to training now?', True) then begin
      NewModel := True;
      ParamsNeedCopyToDevice := True;
      WorkingName := CurrentBaseName;

      RunTrain(WModelParams, WModelState, TokenizedCorpus);

      if TrainSuccess then
        MaybeSaveModel;

      if TrainSuccess and AskYesNo('Proceed to inference?', True) then
        RunInfer(WModelParams, WModelState);
    end;
  end;
end;

// Main workflow: T = Train
procedure DoTrain;
var
  ModelChoice: string;
begin
  Writeln;
  Writeln('--- Train ---');
  Writeln('Required: token list and matching symbol table.');
  Writeln('You may start a new model or resume from a saved model.');
  Writeln;

  if Length(TokenizedCorpus) = 0 then begin
    if not LoadTokenListPrompt then
      Exit;
  end
  else if AskYesNo('Use token list already in memory?', True) then begin
    Writeln('Using in-memory token list. Tokens = ', Length(TokenizedCorpus));
  end
  else
    if not LoadTokenListPrompt then Exit;

  if Length(SymbolTable) < MinSymbols then
    if not LoadSymbolTablePrompt then Exit
  else begin
    nSymbols := Length(SymbolTable);
    nVocab := nSymbols;
    Writeln('Using symbol table already in memory. Symbols = ', nSymbols);
  end;

  ModelChoice := AskChoice('Model: N = new model, R = resume/load saved model', 'N/R');

  if ModelChoice = 'R' then begin
    if not LoadModelPrompt then Exit;

    if nVocab <> nSymbols then begin
      Writeln('Warning: loaded model nVocab = ', nVocab, ' but current symbol table nSymbols = ', nSymbols, '.');
      Writeln('For resumed training these should match.');

      if not AskYesNo('Continue anyway?', False) then Exit;
    end;
  end
  else begin
    NewModel := True;
    ParamsNeedCopyToDevice := True;
    nVocab := nSymbols;
  end;

  WorkingName := CurrentBaseName;
  RunTrain(WModelParams, WModelState, TokenizedCorpus);

  if TrainSuccess then begin
    MaybeSaveModel;

    if AskYesNo('Proceed to inference?', False) then
      RunInfer(WModelParams, WModelState);
  end;
end;

// Main workflow: I = Infer
procedure DoInfer;
begin
  Writeln;
  Writeln('--- Infer ---');
  Writeln('Required: model and matching symbol table.');
  Writeln;

  // Establish tokenizer and symbol-table information first.
  if not LoadSymbolTablePrompt then
    Exit;

  SetTokenizerMode(WesTokenizer);

  // LoadModel now restores the saved model nVocab.
  if not LoadModelPrompt then Exit;

  if nVocab <> nSymbols then begin
    Writeln('Vocabulary mismatch. Inference aborted.');
    Writeln('Model nVocab       = ', nVocab);
    Writeln('Symbol table count = ', nSymbols);
    Exit;
  end;

  ParamsNeedCopyToDevice := True;
  RunInfer(WModelParams, WModelState);
end;

// Main workflow: J = Join symbol tables.
procedure DoJoinSymbolTables;
var
  UChoice, S: string;
begin
  Writeln;
  Writeln('--- Utilities ---');
  Writeln('J: Join two symbol tables');
  Writeln('X: Return to main menu');
  Writeln;

  UChoice := AskChoice('Utility', 'J/X');

  case UChoice of
    'J': begin
      MergeSymbolTables(CombinedSymbolTable);

      Write('Output combined symbol table name, blank for combined.sym: ');
      Readln(S);

      SymbolFileName := MakeOutputFileName(S, SymbolDir, 'joined', '.sym');
      SaveSymbolTable(SymbolFileName, CombinedSymbolTable);

      Writeln('File ', SymbolFileName, ' successfully saved.');
    end;
  end;
  ResetWesTrie;
end;

// Folder utilities.
procedure ShowWorkFolders;
begin
  Writeln;
  Writeln('--- Current Work Folders ---');
  Writeln('ExistingWorkRoot = ', ExistingWorkRoot);
  Writeln('WorkingDir       = ', WorkingDir);
  Writeln('WorkRoot         = ', WorkRoot);
  Writeln('CorpusDir        = ', CorpusDir);
  Writeln('SymbolDir        = ', SymbolDir);
  Writeln('MergeDir         = ', MergeDir);
  Writeln('TokenDir         = ', TokenDir);
  Writeln('ModelDir         = ', ModelDir);
  Writeln('LogDir           = ', LogDir);
  Writeln('ListDir          = ', ListDir);
  Writeln('ScratchDir       = ', ScratchDir);
end;

function CountFilesInDir(const DirName: string): Integer;
var
  SR: TSearchRec;
  Path: string;
begin
  Result := 0;

  if Trim(DirName) = '' then Exit;

  Path := IncludeTrailingPathDelimiter(DirName);

  if SysUtils.FindFirst(Path + '*.*', faAnyFile, SR) = 0 then begin
    try
      repeat
        if (SR.Name <> '.') and (SR.Name <> '..') then
          if (SR.Attr and faDirectory) = 0 then
            Inc(Result);
      until SysUtils.FindNext(SR) <> 0;
    finally
      SysUtils.FindClose(SR);
    end;
  end;
end;

procedure ShowWorkFolderFileCounts;
begin
  Writeln;
  Writeln('--- Work Folder File Counts ---');
  Writeln('CorpusDir  : ', CountFilesInDir(CorpusDir));
  Writeln('SymbolDir  : ', CountFilesInDir(SymbolDir));
  Writeln('MergeDir   : ', CountFilesInDir(MergeDir));
  Writeln('TokenDir   : ', CountFilesInDir(TokenDir));
  Writeln('ModelDir   : ', CountFilesInDir(ModelDir));
  Writeln('LogDir     : ', CountFilesInDir(LogDir));
  Writeln('ScratchDir : ', CountFilesInDir(ScratchDir));
end;

procedure ListWorkFolderSubfolders;
begin
  Writeln;
  Writeln('--- Work Folder Subfolders ---');

  Writeln('corpus  : ', CorpusDir);
  Writeln('symbols : ', SymbolDir);
  Writeln('merges  : ', MergeDir);
  Writeln('tokens  : ', TokenDir);
  Writeln('models  : ', ModelDir);
  Writeln('logs    : ', LogDir);
  Writeln('scratch : ', ScratchDir);
end;

procedure ChangeExistingWorkRoot;
var
  NewDir, OldDir: string;
begin
  Writeln;
  Writeln('Current predefined-work root: ', ExistingWorkRoot);
  Write('Enter new predefined-work root, blank to cancel: ');
  Readln(NewDir);

  NewDir := Trim(NewDir);

  if NewDir = '' then begin
    Writeln('Predefined-work root unchanged.');
    Exit;
  end;

  OldDir := ExistingWorkRoot;
  ExistingWorkRoot := NewDir;
  NormalizeExistingWorkRoot;

  if not DirectoryExists(ExistingWorkRoot) then begin
    if AskYesNo('Folder does not exist. Create it?', True) then begin
      if not ForceDirectories(ExistingWorkRoot) then begin
        Writeln('Unable to create folder: ', ExistingWorkRoot);
        ExistingWorkRoot := OldDir;
        Exit;
      end;
    end
    else begin
      ExistingWorkRoot := OldDir;
      Writeln('Predefined-work root unchanged.');
      Exit;
    end;
  end;

  Writeln('Predefined-work root changed to: ', ExistingWorkRoot);
  Writeln('The current active work folder remains: ', WorkRoot);
end;

procedure ChangeWorkFolder;
var
  NewDir: string;
begin
  Writeln;
  Write('Enter new work folder, blank to cancel: ');
  Readln(NewDir);

  NewDir := Trim(NewDir);

  if NewDir = '' then begin
    Writeln('Work folder unchanged.');
    Exit;
  end;

  WorkingDir := ResolveWorkFolder(NewDir);
  InitWorkFolders(WorkingDir);

  Writeln('Work folder changed.');
  ShowWorkFolders;
end;

procedure DoFolderUtilities;
var
  Choice: string;
begin
  repeat
    Writeln;
    Writeln('--- File/Folder Utilities ---');
    Writeln('C: Change current work folder');
    Writeln('E: Change existing/predefined work root');
    Writeln('L: List work folder subfolders');
    Writeln('S: Show current folder settings');
    Writeln('D: Show file counts in work folders');
    Writeln('O: Open work folder in Explorer');
    Writeln('X: Return to main menu');
    Writeln;

    Choice := AskChoice('Folder command', 'C/E/L/S/D/O/X');

    case Choice of
      'C': ChangeWorkFolder;
      'E': ChangeExistingWorkRoot;
      'L': ListWorkFolderSubfolders;
      'S': ShowWorkFolders;
      'D': ShowWorkFolderFileCounts;
      'O': OpenWorkFolderInExplorer;
    end;

    if Choice <> 'X' then Pause;

  until Choice = 'X';
end;

// Main workflow for Bela.
procedure DoBelaModel;
begin
  Writeln;
  Writeln('--- Bela Corpus ---');

  SelectExistingWork('bela', 'bela');
  TokenizedCorpusPresent := False;
  SymbolTablePresent := False;
  ModelPresent := False;

  CorpusFileName := ResolveInputFile('bela.txt', CorpusDir);
  SymbolFileName := ResolveInputFile('bela.sym', SymbolDir);

  if not FileExists(CorpusFileName) then begin
    Writeln('File not found: ', CorpusFileName);
    Exit;
  end;

  if not FileExists(SymbolFileName) then begin
    Writeln('File not found: ', SymbolFileName);
    Exit;
  end;

  ReadFileBytes(CorpusFileName, Corpus);
  nCorpus := Length(Corpus);
  FromSymbolTable := True;

  SetLength(CorpusFileNames, 1);
  CorpusFileNames[0] := CorpusFileName;

  LoadSymbolTable(SymbolFileName, SymbolTable);
  ResetWesTrie;

  nSymbols := Length(SymbolTable);
  SetTokenizerMode(WesTokenizer);

  NewModel := True;
  ParamsNeedCopyToDevice := True;

  RunWesTokenizeNoAutoSave(Corpus, TokenizedCorpus);

  PadToSeqMultiple(TokenizedCorpus, SeqLen);
  nTokenizedCorpus := Length(TokenizedCorpus);
  TokenizedCorpusPresent := True;
  SymbolTablePresent := True;
  ModelPresent := False;

  Writeln('Bela tokenization complete. Tokens = ', Length(TokenizedCorpus), '; Symbols = ', nSymbols, '.');

  if AskYesNo('Save refreshed Bela token list?', False) then
    SaveCurrentTokenListDefault;

  if AskYesNo('Proceed to training?', True) then begin
    WorkingName := CurrentBaseName;
    RunTrain(WModelParams, WModelState, TokenizedCorpus);

    if TrainSuccess then begin
      MaybeSaveModel;

      if AskYesNo('Proceed to inference?', False) then
        RunInfer(WModelParams, WModelState);
    end;
  end;
end;

// Main work flow: Damned Thing.
procedure DoDamnedThingModel;
begin
  Writeln;
  Writeln('--- Damned Thing Corpus ---');

  SelectExistingWork('dt327', 'dt327');
  TokenizedCorpusPresent := False;
  SymbolTablePresent := False;
  ModelPresent := False;

  TokenFileName := ResolveInputFile('dt327.tok', TokenDir);
  SymbolFileName := ResolveInputFile('dt327.sym', SymbolDir);

  if not FileExists(TokenFileName) then begin
    Writeln('File not found: ', TokenFileName);
    Exit;
  end;

  if not FileExists(SymbolFileName) then begin
    Writeln('File not found: ', SymbolFileName);
    Exit;
  end;

  IOHandler.LoadTokenList(TokenFileName, TokenizedCorpus);
  PadToSeqMultiple(TokenizedCorpus, SeqLen);
  nTokenizedCorpus := Length(TokenizedCorpus);
  TokenizedCorpusPresent := True;
  SymbolTablePresent := True;
  ModelPresent := False;

  FromSymbolTable := True;
  LoadSymbolTable(SymbolFileName, SymbolTable);
  ResetWesTrie;

  nSymbols := Length(SymbolTable);
  SetTokenizerMode(WesTokenizer);

  NewModel := True;
  ParamsNeedCopyToDevice := True;

  Writeln('Damned Thing data loaded. Tokens = ', Length(TokenizedCorpus),
    '; Symbols = ', nSymbols, '.');

  if AskYesNo('Proceed to training?', True) then begin
    WorkingName := CurrentBaseName;
    RunTrain(WModelParams, WModelState, TokenizedCorpus);

    if TrainSuccess then begin
      MaybeSaveModel;

      if AskYesNo('Proceed to inference?', False) then
        RunInfer(WModelParams, WModelState);
    end;
  end;
end;

procedure DoResumeBestModel(const ModelTitle, FolderName, DataBaseName: string);
begin
  Writeln;
  Writeln('--- ', ModelTitle, ' ---');

  SelectExistingWork(FolderName, DataBaseName);

  TokenizedCorpusPresent := False;
  SymbolTablePresent := False;
  ModelPresent := False;

  TokenFileName := TokenDir + DataBaseName + '.tok';
  SymbolFileName := SymbolDir + DataBaseName + '.sym';
  ModelFileName := ModelDir + 'weschat_best.model';

  // Verify all required files before replacing in-memory data.
  if not FileExists(TokenFileName) then begin
    Writeln('Token file not found: ', TokenFileName);
    Exit;
  end;

  if not FileExists(SymbolFileName) then begin
    Writeln('Symbol table not found: ', SymbolFileName);
    Exit;
  end;

  if not FileExists(ModelFileName) then begin
    Writeln('Model file not found: ', ModelFileName);
    Exit;
  end;

  // Load tokenized corpus.
  SetLength(TokenizedCorpus, 0);
  IOHandler.LoadTokenList(TokenFileName, TokenizedCorpus);

  if Length(TokenizedCorpus) < MinTokens then begin
    Writeln('Token list is too small. Length = ', Length(TokenizedCorpus), '.');
    Exit;
  end;

  PadToSeqMultiple(TokenizedCorpus, SeqLen);
  nTokenizedCorpus := Length(TokenizedCorpus);

  // Load matching Wes symbol table.
  ResetWesTrie;
  SetLength(SymbolTable, 0);

  FromSymbolTable := True;
  LoadSymbolTable(SymbolFileName, SymbolTable);

  if Length(SymbolTable) < MinSymbols then begin
    Writeln('Symbol table is too small. Length = ', Length(SymbolTable), '.');
    Exit;
  end;

  nSymbols := Length(SymbolTable);
  SetTokenizerMode(WesTokenizer);

  // Discard CUDA storage belonging to the preceding model.
  if CudaAllocated or (CuHandle <> nil) then
    EndCuda(WModelParams, WModelState);

  // Load saved parameters and model information.
  if not LoadModel(ModelFileName, WModelParams) then begin
    Writeln('Model not loaded: ', ModelFileName);
    Exit;
  end;

  NewModel := False;
  ParamsNeedCopyToDevice := True;

  // LoadModel should have restored nVocab.
  if nVocab <> nSymbols then begin
    Writeln('Vocabulary mismatch. Training aborted.');
    Writeln('Model nVocab       = ', nVocab);
    Writeln('Symbol table count = ', nSymbols);
    Exit;
  end;

  if nVocab > DimVocab then begin
    Writeln('Model vocabulary exceeds DimVocab. Training aborted.');
    Writeln('nVocab   = ', nVocab);
    Writeln('DimVocab = ', DimVocab);
    Exit;
  end;

  TokenizedCorpusPresent := True;
  SymbolTablePresent := True;
  ModelPresent := True;

  Writeln;
  Write('Loaded Token list = ', TokenFileName);
  Write('; Symbol table = ', SymbolFileName);
  Write('; Model = ', ModelFileName);
  Write('; Tokens = ', nTokenizedCorpus);
  Write('; Symbols = ', nSymbols);
  Write('; nVocab = ', nVocab, '.');

  if not AskYesNo('Proceed to resumed training?', True) then Exit;

  // Preserve weschat_best.model and save new best models under the corpus name.
  WorkingName := CurrentBaseName;

  Writeln('Loaded checkpoint remains: ', ModelFileName);
  Writeln('New automatic best model: ', ModelDir + WorkingName + '_best.model');

  RunTrain(WModelParams, WModelState, TokenizedCorpus);

  if TrainSuccess then begin
    if AskYesNo('Save an additional model?', False) then
      MaybeSaveModel;

    if AskYesNo('Proceed to inference?', False) then
      RunInfer(WModelParams, WModelState);
  end;
end;

procedure DoGibbon731bModel;
begin
  DoResumeBestModel('Gibbon 731b Best Model', 'gibbon731b', 'gibbon');
end;

procedure DoLocke730Model;
begin
  DoResumeBestModel('Locke 730 Best Model', 'locke730', 'locke');
end;

procedure DoChurchill801Model;
begin
  DoResumeBestModel('Churchill 801 Best Model', 'churchill801', 'churchill');
end;

procedure DoPredefinedWork;
var
  TChoice: string;
begin
  repeat
    Writeln;
    Writeln('--- Predefined Corpora and Models ---');
    Writeln('B: Train a new model on the Bela corpus');
    Writeln('D: Train a new model on the Damned Thing token list');
    Writeln('G: Resume the Gibbon 731b best model');
    Writeln('C: Resume the Churchill 801 best model');
    Writeln('L: Resume the Locke 730 best model');
    Writeln('X: Return to main menu');
    Writeln;

    TChoice := AskChoice('Selection', 'B/D/G/C/L/X');

    case TChoice of
      'B', 'BELA': DoBelaModel;
      'D', 'DT': DoDamnedThingModel;
      'G', 'GIBBON': DoGibbon731bModel;
      'C', 'CHURCHILL': DoChurchill801Model;
      'L', 'LOCKE': DoLocke730Model;
    end;

  until TChoice = 'X';
end;

// Display / menu.
procedure Options;
begin
  Writeln;
  Writeln('Options:');
  Writeln('  T: Tokenize -- create a token list from a corpus file or a file list.');
  Writeln('     Uses WesTokenize or GPT2Tokenize.');
  Writeln('     WesTokenize may create a symbol table or use an existing one.');
  Writeln;
  Writeln('  R: Train -- train a model on a token list.');
  Writeln('     Requires a token list and matching symbol table.');
  Writeln('     Can start a new model or resume from a saved model.');
  Writeln;
  Writeln('  I: Infer -- run inference.');
  Writeln('     Requires a saved model and matching symbol table.');
  Writeln;
  Writeln('  J: Join symbol tables.');
  Writeln('     Requires two symbol tables.');
  Writeln;
  Writeln('  M: Do predefined work -- train corpora or resume models: Bela, Damned Thing, Gibbon, Locke, or Churchill.');
  Writeln('  F: File/folder utilities.');
  Writeln('  P: Program information.');
  Writeln('  H: Help and options.');
  Writeln('  X: Exit.');
  Writeln;
end;

procedure Help;
begin
  Options;

  Writeln('Work folder layout:');
  Writeln('  ', CorpusDir,  '   corpus input files');
  Writeln('  ', SymbolDir,  '   .sym files');
  Writeln('  ', MergeDir,   '   .mer files');
  Writeln('  ', TokenDir,   '   .tok files');
  Writeln('  ', ModelDir,   '   saved models');
  Writeln('  ', LogDir,     '   logs');
  Writeln('  ', ListDir,    '   file lists');
  Writeln;

  Writeln('Debug / display toggles available:');
  Writeln('  VTO / NVTO: VerboseTokenize on/off');
  Writeln('  DC / NDC:   DisplayCorpus on/off');
  Writeln('  DTW / NDTW: DisplayTokenWork on/off');
  Writeln('  DMW / NDMW: DisplayMergeWork on/off');
  Writeln('  DV / NDV:   DisplayVerification on/off');
  Writeln('  DEBR / NDEBR: DisplayEachByteRead on/off');
  Writeln('  DW / NDW: DisplayWindow on/off');
  Writeln('  DNP / DP: DoNotPause on/off');
  Writeln('  SF / NSF: SaveFiles on/off');
  Writeln;
  Writeln('Training display:');
  Writeln('  DE:  Display epochs');
  Writeln('  DS:  Display stages');
  Writeln('  DSS: Display substages');
  Writeln('  ND:  Reduce display');
  Writeln;
  Writeln('Parameters:');
  Writeln('  MM:   Maximum merges');
  Writeln('  PC:   Maximum pair count');
  Writeln('  LR:   Override learning rate');
  Writeln('  TEMP: Temperature');
  Writeln;
end;

procedure HandleSettingCommand(const Cmd: string);
begin
  case Cmd of
    'VTO': begin
      VerboseTokenize := True;
      Writeln('Verbose tokenize is ', VerboseTokenize);
    end;
    'NVTO':  begin
      VerboseTokenize := False;
      Writeln('Verbose tokenize is ', VerboseTokenize);
    end;
    'DC':    begin
      Writeln('Display corpus is ', DisplayCorpus);
      DisplayCorpus := True;
    end;
    'NDC': begin
      DisplayCorpus := False;
      Writeln('Display corpus is ', DisplayCorpus);
    end;
    'DTW':    begin
      Writeln('Display token work is ', DisplayTokenWork);
      DisplayTokenWork := True;
    end;
    'NDTW': begin
      DisplayTokenWork := False;
      Writeln('Display token work is ', DisplayTokenWork);
    end;
    'DMW': begin
      DisplayMergeWork := True;
      Writeln('Display merge work is ', DisplayMergeWork);
    end;
    'NDMW': begin
      DisplayMergeWork := False;
      Writeln('Display merge work is ', DisplayMergeWork);
    end;
    'DTV': begin
      DisplayTokenVerification := True;
      Writeln('Display token verification is ', DisplayTokenVerification);
    end;
    'NDTV': begin
      DisplayTokenVerification := False;
      Writeln('Display token verification is ', DisplayTokenVerification);
    end;
    'DEBR':  begin
      DisplayEachByteRead := True;
      Writeln('Display each byte read is', DisplayEachByteRead);
    end;
    'NDEBR':  begin
      DisplayEachByteRead := False;
      Writeln('Display each byte read is', DisplayEachByteRead);
    end;
    'VTR':   begin
      VerboseTransform := True;
      DisplayStage := True;
      Writeln('Verbose transform is ', VerboseTransform);
    end;
    'NVTR':   begin
      VerboseTransform := False;
      Writeln('Verbose transform is ', VerboseTransform);
    end;
    'DE': begin
      DisplayEpoch := True;
      DisplayStage := False;
      DisplaySubstage := False;
    end;
    'VI':   begin
      VerboseInfer := True;
      Writeln('Verbose infer is ', VerboseInfer);
    end;
    'NVI':   begin
      VerboseInfer := False;
      Writeln('Verbose infer is ', VerboseInfer);
    end;
    'DS': begin
      DisplayStage := True;
      DisplayEpoch := True;
      DisplaySubstage := False;
    end;
    'DSS': begin
      DisplaySubstage := True;
      DisplayStage := True;
      DisplayEpoch := True;
    end;
    'ND': begin
      DisplayEpoch := False;
      DisplayStage := False;
      DisplaySubstage := False;
    end;
    'DW': begin
      DisplayWindow := True;
      Writeln('Display window is ', DisplayWindow);
    end;
    'NDW': begin
      DisplayWindow := False;
      Writeln('Display window is ', DisplayWindow);
    end;
    'DNP': begin
      DoNotPause := True;
      Writeln('Do not pause is', DoNotPause);
    end;
    'DP': begin
      DoNotPause := False;
      Writeln('Do not pause is', DoNotPause);
    end;
    'SF': begin
      SaveFiles := True;
      Writeln('Save files is ', SaveFiles);
    end;
    'NSF': begin
      SaveFiles := False;
      Writeln('Save files is ', SaveFiles);
    end;
    'TEMP': begin
      Write('Inference temperature: ');
      Readln(ITemperature);
    end;
    'LR': begin
      Write('Override learning rate: ');
      Readln(OverrideLearningRate);
    end;
    'MM': begin
      Write('Maximum merges: ');
      Readln(MaxMerges);
    end;
    'PC': begin
      Write('Maximum pair count: ');
      Readln(MaxPairCount);
    end;
  end;
end;

// Program startup
begin
  SetMultiByteConversionCodePage(CP_UTF8);
  SetMultiByteRTLFileSystemCodePage(CP_UTF8);

  SetConsoleOutputCP(CP_UTF8);
  SetConsoleCP(CP_UTF8);

  Writeln('WesChat, Version 1.2, begun January 10, 2026, by Wesley R. Parsons.');
  Writeln;

  ReportKeyVariables;
  Writeln;

  NormalizeExistingWorkRoot;

  Writeln('Work folders are normally created under ', ExistingWorkRoot);
  Write('Enter work folder name, blank for WesChatWork: ');
  Readln(WorkingDir);

  WorkingDir := ResolveWorkFolder(WorkingDir);
  InitWorkFolders(WorkingDir);

  Writeln('Work folder: ', WorkRoot);

  Options;

  while True do begin
    Write('W>');
    Readln(Ch);
    Ch := UpperCase(Trim(Ch));

    case Ch of
      'T': DoTokenize;
      'R': DoTrain;
      'I': DoInfer;
      'J': DoJoinSymbolTables;
      'M': DoPredefinedWork;
      'F': DoFolderUtilities;
      'P': ReportProgramInfo;
      'H': Help;
      'X', 'EXIT': Break;

      'VTO', 'NVTO', 'DC', 'NDC', 'DTW', 'NDTW',
      'DMW', 'NDMW', 'DTV', 'NDTV', 'DEBR', 'NDEBR',
      'VTR', 'NVTR', 'VI', 'NVI',
      'DE', 'DS', 'DSS', 'ND',
      'DW', 'NDW', 'DNP', 'DP', 'SF', 'NSF',
      'TEMP', 'LR', 'MM', 'PC':
        HandleSettingCommand(Ch);

      else
        Writeln('Invalid input. Enter H for help.');
    end;
  end;

  // Stop cuda.
  if CudaAllocated or (CuHandle <> nil) then
    EndCuda(WModelParams, WModelState);

  // Free vocab.
  if Assigned(Vocab) then
    Vocab.Free;
end.
