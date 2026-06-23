program WesChat;

{$mode ObjFPC}{$H+}{$I proprietary.txt}

{ WesChat, Version 1.2, begun January 10, 2026, by Wesley R. Parsons, wespar@bellouth.net, www.wesparsons.com }
{ Note: Edited 6/22/2026 8 pm -- working from WesChat12 on OneDrive }
{        Input Train        Input Query        Output
 Raw                        QueryString
 Bytes   Corpus             QueryCorpus
 Token   TokenizedCorpus    QueryTokenized     QueryOutput }
 {For tokenizining, need tokenized corpus.                     CorpusPresent (now TokSuccessful)
  For training, need symbol table for WesChat.                 SMTablePresent
    (already have symbol and merge tables for GPT2).
    and token list.                                            TokenizedCorpusPresent
  For inference, need param model, and symbol/merge table(s).  ModelPresent }


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

  MinSymbols: Integer = 50;
  MinTokens: Integer = 50;
  MinCorpus: Integer = 50;

  Ch: string;
  CombinedSymbolTable: TSymbolTable;


// ------------------------------------------------------------
// General helpers
// ------------------------------------------------------------

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


function RequireExistingFile(const FileName: string): Boolean;
begin
  Result := FileExists(FileName);

  if not Result then
    Writeln('File not found: ', FileName, '.');
end;


function RequireMinFileSize(const FileName: string; MinSize: Integer): Boolean;
begin
  Result := FileSize(FileName) >= MinSize;

  if not Result then
    Writeln('File too small: ', FileName, '. Size=', FileSize(FileName),
      ' minimum=', MinSize, '.');
end;


// Create and name directory and file for saving.
procedure LogFile(const Eponym: string);
var
  SaveOut: Text;
begin
  WorkingDir := ChangeFileExt(Eponym, '') + FormatDateTime('yyyy-mm-dd_hhnnss', Now);
  WorkingName := WorkingDir;

  CreateDir(WorkingDir);
  ChDir(WorkingDir);

  SaveOut := Output;

  Assign(Output, WorkingName + '.log');
  ReWrite(Output);
  ReportInfo;

  Close(Output);
  Output := SaveOut;

  ChDir('..');
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
procedure ProcessFileList(var ListFile: string; var Corpus: TBVector);
var
  F: TextFile;
  Line: string;
  OneCorpus: TBVector;
  Count: Integer;
begin
  MultipleFileName := EmptyStr;

  Write('Enter name of file list: ');
  Readln(ListFile);

  if not RequireExistingFile(ListFile) then
    Exit;

  AssignFile(F, ListFile);
  Reset(F);

  Count := 0;
  FromSymbolTable := False;
  SetLength(Corpus, 0);
  SetLength(CorpusFileNames, 0);

  while not EOF(F) do begin
    ReadLn(F, Line);
    Line := Trim(Line);

    if Line = '' then
      Continue;

    if not FileExists(Line) then begin
      Writeln('  File not found: ', Line, '.');
      Continue;
    end;

    if FileSize(Line) < MinCorpus then begin
      Writeln('  Corpus too small, skipping: ', Line);
      Continue;
    end;

    if (Count = 0) and SaveFiles then
      LogFile('Mult' + ListFile);

    ReadFileBytes(Line, OneCorpus);

    SetLength(CorpusFileNames, Count + 1);
    CorpusFileNames[Count] :=
      Line + '   ' + IntToStr(FileSize(Line)) + ' bytes   ' +
      DateTimeToStr(FileDateToDateTime(FileAge(Line)));

    Corpus := Concat(Corpus, OneCorpus);
    nCorpus := Length(Corpus);

    Writeln('  File processed: ', Line,
      '; corpus bytes read: ', Length(OneCorpus), '.');
    Writeln('  Total bytes read: ', Length(Corpus), '.');

    Inc(Count);
  end;

  CloseFile(F);

  Writeln('Combined corpus length = ', Length(Corpus));
  nCorpus := Length(Corpus);
end;


// ------------------------------------------------------------
// Loading helpers
// ------------------------------------------------------------

function LoadSymbolTablePrompt: Boolean;
begin
  Result := False;

  Write('Input symbol table file name: ');
  Readln(SymbolFileName);

  if not RequireExistingFile(SymbolFileName) then
    Exit;

  FromSymbolTable := True;
  LoadSymbolTable(SymbolFileName, SymbolTable);

  if Length(SymbolTable) < MinSymbols then begin
    Writeln('Too few symbols found. Length(SymbolTable)=', Length(SymbolTable));
    Exit;
  end;

  nSymbols := Length(SymbolTable);
  nVocab := nSymbols;

  Result := True;
end;


function LoadTokenListPrompt: Boolean;
begin
  Result := False;

  Write('Enter token list file name: ');
  Readln(TokenFileName);

  if not RequireExistingFile(TokenFileName) then
    Exit;

  IOHandler.LoadTokenList(TokenFileName, TokenizedCorpus);

  if Length(TokenizedCorpus) < MinTokens then begin
    Writeln('Token list too small. Length=', Length(TokenizedCorpus));
    Exit;
  end;

  PadToSeqMultiple(TokenizedCorpus, SeqLen);
  nTokenizedCorpus := Length(TokenizedCorpus);

  Result := True;
end;


function LoadModelPrompt: Boolean;
begin
  Result := False;

  Write('Enter model file name: ');
  Readln(ModelFileName);

  if not RequireExistingFile(ModelFileName) then
    Exit;

  if CudaAllocated then
    MDeallocateCublas(WModelParams, WModelState);

  if LoadModel(ModelFileName, WModelParams) then begin
    Writeln('File ', ModelFileName, ' loaded.');
    NewModel := False;
    ParamsNeedCopyToDevice := True;
    Result := True;
  end
  else begin
    Writeln('File not loaded.');
  end;
end;


// ------------------------------------------------------------
// Save helpers
// ------------------------------------------------------------

procedure MaybeSaveTokenList;
begin
  if Length(TokenizedCorpus) = 0 then
    Exit;

  if AskYesNo('Save token list?', False) then begin
    Write('Output token list file name: ');
    Readln(TokenFileName);
    SaveTokenList(TokenizedCorpus, TokenFileName);
  end;
end;


procedure MaybeSaveSymbolTable;
begin
  if Length(SymbolTable) = 0 then
    Exit;

  if AskYesNo('Save symbol table?', False) then begin
    Write('Output symbol table file name: ');
    Readln(SymbolFileName);
    SaveSymbolTable(SymbolFileName, SymbolTable);
  end;
end;


procedure MaybeSaveModel;
begin
  if AskYesNo('Save model?', False) then begin
    Write('Output model file name: ');
    Readln(ModelFileName);

    if SaveModel(ModelFileName, WModelParams) then
      Writeln('File ', ModelFileName, ' successfully saved.')
    else
      Writeln('File not saved.');
  end;
end;


// ------------------------------------------------------------
// Main workflow: K = Tokenize
// ------------------------------------------------------------

procedure TokenizeWithWes;
var
  SourceChoice, SymbolChoice: string;
begin
  SetLength(TokenizedCorpus, 0);

  SourceChoice := AskChoice(
    'Corpus source: F = one file, L = list of corpus file names',
    'F/L');

  if SourceChoice = 'L' then begin
    ProcessFileList(ListFile, Corpus);

    if Length(Corpus) < MinCorpus then begin
      Writeln('Combined corpus too small. Aborting tokenization.');
      Exit;
    end;
  end
  else begin
    Write('Enter corpus file name: ');
    Readln(CorpusFileName);

    if not RequireExistingFile(CorpusFileName) then
      Exit;

    if not RequireMinFileSize(CorpusFileName, MinCorpus) then
      Exit;

    ReadFileBytes(CorpusFileName, Corpus);
    nCorpus := Length(Corpus);

    SetLength(CorpusFileNames, 1);
    CorpusFileNames[0] :=
      CorpusFileName + '   ' + IntToStr(FileSize(CorpusFileName)) +
      ' bytes   ' + DateTimeToStr(FileDateToDateTime(FileAge(CorpusFileName)));

    if SaveFiles then
      LogFile(CorpusFileName);
  end;

  SymbolChoice := AskChoice(
    'Symbol table: C = create from corpus, S = load existing symbol table',
    'C/S');

  if SymbolChoice = 'S' then begin
    if not LoadSymbolTablePrompt then
      Exit;
  end
  else begin
    FromSymbolTable := False;
    nSymbols := 0;
    nVocab := 0;
    SetLength(SymbolTable, 0);

    RunSymbolize(Corpus);

    nSymbols := Length(SymbolTable);
    nVocab := nSymbols;

    if nSymbols < MinSymbols then begin
      Writeln('Too few symbols found after symbolization. nSymbols=', nSymbols);
      Exit;
    end;

    MaybeSaveSymbolTable;
  end;

  Tokenizer := WesTokenizer;
  SaveTokenizationFiles := True;
  RunWesTokenize(Corpus, TokenizedCorpus);
  PadToSeqMultiple(TokenizedCorpus, SeqLen);
  nTokenizedCorpus := Length(TokenizedCorpus);

  Writeln('Tokenization complete. Tokens=', Length(TokenizedCorpus),
    ' Symbols=', nSymbols, '.');

  MaybeSaveTokenList;
end;


procedure TokenizeGPTSingleFile;
begin
  SetLength(TokenizedCorpus, 0);

  Write('Enter corpus file name: ');
  Readln(CorpusFileName);

  if not RequireExistingFile(CorpusFileName) then
    Exit;

  if not RequireMinFileSize(CorpusFileName, MinCorpus) then
    Exit;

  ReadFileBytes(CorpusFileName, Corpus);
  nCorpus := Length(Corpus);

  SetLength(CorpusFileNames, 1);
  CorpusFileNames[0] :=
    CorpusFileName + '   ' + IntToStr(FileSize(CorpusFileName)) +
    ' bytes   ' + DateTimeToStr(FileDateToDateTime(FileAge(CorpusFileName)));

  if SaveFiles then
    LogFile(CorpusFileName);

  Tokenizer := GPT2Tokenizer;
  RunGPT2Tokenize(CorpusFileName, TokenizedCorpus);
  PadToSeqMultiple(TokenizedCorpus, SeqLen);
  nTokenizedCorpus := Length(TokenizedCorpus);

  Writeln('GPT tokenization complete. Tokens=', Length(TokenizedCorpus), '.');

  MaybeSaveTokenList;
end;


procedure TokenizeGPTFileList;
var
  F: TextFile;
  Line: string;
  OneTokens: TIVector;
  Count: Integer;
begin
  SetLength(TokenizedCorpus, 0);

  Write('Enter name of file list: ');
  Readln(ListFile);

  if not RequireExistingFile(ListFile) then
    Exit;

  AssignFile(F, ListFile);
  Reset(F);

  Count := 0;
  SetLength(CorpusFileNames, 0);

  while not EOF(F) do begin
    Readln(F, Line);
    Line := Trim(Line);

    if Line = '' then
      Continue;

    if not FileExists(Line) then begin
      Writeln('  File not found: ', Line, '.');
      Continue;
    end;

    if FileSize(Line) < MinCorpus then begin
      Writeln('  Corpus too small, skipping: ', Line);
      Continue;
    end;

    SetLength(OneTokens, 0);
    RunGPT2Tokenize(Line, OneTokens);
    AppendTokens(TokenizedCorpus, OneTokens);

    SetLength(CorpusFileNames, Count + 1);
    CorpusFileNames[Count] :=
      Line + '   ' + IntToStr(FileSize(Line)) + ' bytes   ' +
      DateTimeToStr(FileDateToDateTime(FileAge(Line)));

    Inc(Count);

    Writeln('  GPT-tokenized: ', Line,
      '; tokens added=', Length(OneTokens),
      '; total tokens=', Length(TokenizedCorpus), '.');
  end;

  CloseFile(F);

  Tokenizer := GPT2Tokenizer;
  PadToSeqMultiple(TokenizedCorpus, SeqLen);
  nTokenizedCorpus := Length(TokenizedCorpus);

  Writeln('GPT list tokenization complete. Tokens=', Length(TokenizedCorpus), '.');

  MaybeSaveTokenList;
end;


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
    SourceChoice := AskChoice(
      'Corpus source: F = one file, L = list of corpus file names',
      'F/L');

    if SourceChoice = 'L' then
      TokenizeGPTFileList
    else
      TokenizeGPTSingleFile;
  end
  else begin
    TokenizeWithWes;
  end;

  if Length(TokenizedCorpus) > 0 then begin
    if AskYesNo('Proceed to training now?', False) then begin
      RunTrain(WModelParams, WModelState, TokenizedCorpus);
      if TrainSuccess then
        MaybeSaveModel;

      if TrainSuccess and AskYesNo('Proceed to inference?', False) then
        RunInfer(WModelParams, WModelState);
    end;
  end;
end;


// ------------------------------------------------------------
// Main workflow: T = Train
// ------------------------------------------------------------

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
    Writeln('Using in-memory token list. Tokens=', Length(TokenizedCorpus));
  end
  else begin
    if not LoadTokenListPrompt then
      Exit;
  end;

  if Length(SymbolTable) < MinSymbols then begin
    if not LoadSymbolTablePrompt then
      Exit;
  end
  else begin
    nSymbols := Length(SymbolTable);
    nVocab := nSymbols;
    Writeln('Using symbol table already in memory. Symbols=', nSymbols);
  end;

  ModelChoice := AskChoice('Model: N = new model, R = resume/load saved model', 'N/R');

  if ModelChoice = 'R' then begin
    if not LoadModelPrompt then
      Exit;

    if nVocab <> nSymbols then begin
      Writeln('Warning: loaded model nVocab=', nVocab,
        ' but current symbol table nSymbols=', nSymbols, '.');
      Writeln('For resumed training these should match.');
      if not AskYesNo('Continue anyway?', False) then
        Exit;
    end;
  end
  else begin
    NewModel := True;
    nVocab := nSymbols;
  end;

  RunTrain(WModelParams, WModelState, TokenizedCorpus);

  if TrainSuccess then begin
    MaybeSaveModel;

    if AskYesNo('Proceed to inference?', False) then
      RunInfer(WModelParams, WModelState);
  end;
end;


// ------------------------------------------------------------
// Main workflow: I = Infer
// ------------------------------------------------------------

procedure DoInfer;
begin
  Writeln;
  Writeln('--- Infer ---');
  Writeln('Required: model and matching symbol table.');
  Writeln;

  if not LoadModelPrompt then
    Exit;

  if not LoadSymbolTablePrompt then
    Exit;

  Tokenizer := WesTokenizer;

  ParamsNeedCopyToDevice := True;
  RunInfer(WModelParams, WModelState);
end;


// ------------------------------------------------------------
// Main workflow: U = Utilities
// ------------------------------------------------------------

procedure DoUtilities;
var
  UChoice: string;
begin
  Writeln;
  Writeln('--- Utilities ---');
  Writeln('C: Combine two symbol tables');
  Writeln('X: Return to main menu');
  Writeln;

  UChoice := AskChoice('Utility', 'C/X');

  case UChoice of
    'C': begin
      MergeSymbolTables(CombinedSymbolTable);

      Write('Output combined symbol table name: ');
      Readln(SymbolFileName);

      if SaveFiles then
        LogFile(SymbolFileName);

      SaveSymbolTable(SymbolFileName, CombinedSymbolTable);
      Writeln('File ', SymbolFileName, ' successfully saved.');
    end;
  end;
end;


// ------------------------------------------------------------
// Main workflow: B / DT = Tests
// ------------------------------------------------------------

procedure DoBelaTest;
begin
  Writeln;
  Writeln('--- Bela test ---');

  CorpusFileName := 'bela.txt';
  SymbolFileName := 'bela.sym';

  if not RequireExistingFile(CorpusFileName) then
    Exit;

  if not RequireExistingFile(SymbolFileName) then
    Exit;

  ReadFileBytes(CorpusFileName, Corpus);
  nCorpus := Length(Corpus);
  FromSymbolTable := True;

  SetLength(CorpusFileNames, 1);
  CorpusFileNames[0] := CorpusFileName;

  LoadSymbolTable(SymbolFileName, SymbolTable);
  nSymbols := Length(SymbolTable);
  nVocab := nSymbols;

  if SaveFiles then
    LogFile(CorpusFileName);

  Tokenizer := WesTokenizer;
  SaveTokenizationFiles := True;
  RunWesTokenize(Corpus, TokenizedCorpus);
  PadToSeqMultiple(TokenizedCorpus, SeqLen);
  nTokenizedCorpus := Length(TokenizedCorpus);

  Writeln('Bela tokenization complete. Tokens=', Length(TokenizedCorpus),
    ' Symbols=', nSymbols, '.');

  if AskYesNo('Proceed to training?', True) then begin
    RunTrain(WModelParams, WModelState, TokenizedCorpus);

    if TrainSuccess then begin
      MaybeSaveModel;

      if AskYesNo('Proceed to inference?', False) then
        RunInfer(WModelParams, WModelState);
    end;
  end;
end;


procedure DoDamnedThingTest;
begin
  Writeln;
  Writeln('--- Damned Thing test ---');

  TokenFileName := 'dt327.tok';
  SymbolFileName := 'dt327.sym';

  if not RequireExistingFile(TokenFileName) then
    Exit;

  if not RequireExistingFile(SymbolFileName) then
    Exit;

  IOHandler.LoadTokenList(TokenFileName, TokenizedCorpus);
  PadToSeqMultiple(TokenizedCorpus, SeqLen);
  nTokenizedCorpus := Length(TokenizedCorpus);

  FromSymbolTable := True;
  LoadSymbolTable(SymbolFileName, SymbolTable);
  nSymbols := Length(SymbolTable);
  nVocab := nSymbols;

  Writeln('Damned Thing data loaded. Tokens=', Length(TokenizedCorpus),
    ' Symbols=', nSymbols, '.');

  if AskYesNo('Proceed to training?', True) then begin
    RunTrain(WModelParams, WModelState, TokenizedCorpus);

    if TrainSuccess then begin
      MaybeSaveModel;

      if AskYesNo('Proceed to inference?', False) then
        RunInfer(WModelParams, WModelState);
    end;
  end;
end;


procedure DoTests;
var
  TChoice: string;
begin
  Writeln;
  Writeln('--- Tests ---');
  Writeln('B: Bela corpus');
  Writeln('D: Damned Thing token list');
  Writeln('X: Return to main menu');
  Writeln;

  TChoice := AskChoice('Test', 'B/D/X');

  case TChoice of
    'B': DoBelaTest;
    'D', 'DT': DoDamnedThingTest;
  end;
end;


// ------------------------------------------------------------
// Display / menu
// ------------------------------------------------------------

procedure Options;
begin
  Writeln;
  Writeln('Options:');
  Writeln('  K: Tokenize -- create a token list from a corpus file or a file list.');
  Writeln('     Uses WesTokenize or GPT2Tokenize.');
  Writeln('     WesTokenize may create a symbol table or use an existing one.');
  Writeln;
  Writeln('  T: Train -- train a model on a token list.');
  Writeln('     Requires a token list and matching symbol table.');
  Writeln('     Can start a new model or resume from a saved model.');
  Writeln;
  Writeln('  I: Infer -- run inference.');
  Writeln('     Requires a saved model and matching symbol table.');
  Writeln;
  Writeln('  U: Utilities -- currently combines symbol tables.');
  Writeln;
  Writeln('  B: Tests -- Bela and Damned Thing presets.');
  Writeln;
  Writeln('  H: Help.');
  Writeln('  X: Exit.');
  Writeln;
end;


procedure Help;
begin
  Options;

  Writeln('Debug / display toggles still available:');
  Writeln('  VTO / NVTO: VerboseTokenize on/off');
  Writeln('  DC / NDC:   DisplayCorpus on/off');
  Writeln('  DTW / NDTW: DisplayTokenWork on/off');
  Writeln('  DMW / NDMW: DisplayMergeWork on/off');
  Writeln('  DV / NDV:   DisplayVerification on/off');
  Writeln('  DEBR / NDEBR: DisplayEachByteRead on/off');
  Writeln('  SPST / NSPST: SavePartialSymbolTable on/off');
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
  Writeln('  M:  Maximum merges');
  Writeln('  PC: Maximum pair count');
  Writeln('  LR: Override learning rate');
  Writeln('  TEMP: Temperature');
  Writeln;
end;


procedure HandleSettingCommand(const Cmd: string);
begin
  case Cmd of
    'VTO':   VerboseTokenize := True;
    'NVTO':  VerboseTokenize := False;
    'DC':    DisplayCorpus := True;
    'NDC':   DisplayCorpus := False;
    'DTW':   DisplayTokenWork := True;
    'NDTW':  DisplayTokenWork := False;
    'DMW':   DisplayMergeWork := True;
    'NDMW':  DisplayMergeWork := False;
    'DV':    DisplayVerification := True;
    'NDV':   DisplayVerification := False;
    'DEBR':  DisplayEachByteRead := True;
    'NDEBR': DisplayEachByteRead := False;
    'SPST':  SavePartialSymbolTable := True;
    'NSPST': SavePartialSymbolTable := False;

    'VTR':   VerboseTransform := True;
    'NVTR':  VerboseTransform := False;

    'DE': begin
      DisplayEpoch := True;
      DisplayStage := False;
      DisplaySubstage := False;
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
      DisplaySubstage := True;
    end;

    'DW':    DisplayWindow := True;
    'NDW':   DisplayWindow := False;

    'DNP':   DoNotPause := True;
    'DP':    DoNotPause := False;

    'SF':    SaveFiles := True;
    'NSF':   SaveFiles := False;

    'TEMP': begin
      Write('Temperature: ');
      Readln(Temperature);
    end;

    'LR': begin
      Write('Override learning rate: ');
      Readln(OverrideLearningRate);
    end;

    'M': begin
      Write('Maximum merges: ');
      Readln(MaxMerges);
    end;

    'PC': begin
      Write('Maximum pair count: ');
      Readln(MaxPairCount);
    end;
  end;
end;


// ------------------------------------------------------------
// Program startup
// ------------------------------------------------------------

begin
  SetMultiByteConversionCodePage(CP_UTF8);
  SetMultiByteRTLFileSystemCodePage(CP_UTF8);

  Vocab := TStringList.Create;
  LoadVocab('vocab1.json', Vocab);

  SetConsoleOutputCP(CP_UTF8);
  SetConsoleCP(CP_UTF8);

  Writeln('WesChat, Version 1.2, begun January 10, 2026, by Wesley R. Parsons.');
  Writeln;

  Options;

  while True do begin
    Write('W>');
    Readln(Ch);
    Ch := UpperCase(Trim(Ch));

    case Ch of
      'K': DoTokenize;
      'T': DoTrain;
      'I': DoInfer;
      'U': DoUtilities;
      'B': DoTests;
      'DT': DoDamnedThingTest;
      'H': Help;
      'X': Break;

      'VTO', 'NVTO', 'DC', 'NDC', 'DTW', 'NDTW',
      'DMW', 'NDMW', 'DV', 'NDV', 'DEBR', 'NDEBR',
      'SPST', 'NSPST', 'VTR', 'NVTR',
      'DE', 'DS', 'DSS', 'ND',
      'DW', 'NDW', 'DNP', 'DP', 'SF', 'NSF',
      'TEMP', 'LR', 'M', 'PC':
        HandleSettingCommand(Ch);

      else
        Writeln('Invalid input. Enter H for help.');
    end;
  end;

  // Clean up CUDA.
  if CudaAllocated then
    EndCuda(WModelParams, WModelState);

  Vocab.Free;
end.







{uses
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
  Symbolize,
  SysUtils,
  Train,
  Util,
  WesTokenize,
  Windows;

var
  // Corpus vars.
  Corpus: TBVector;                         // Vector of byte.
  TokenizedCorpus: TIVector;
  // Model vars.
  WModelParams: TWModelParams;              // Parameters.
  WModelState: TWModelState;                // State.
  // Saving and loading vars.
  CorpusFileName, SymbolFileName,           // File names.
    TokenFileName, ModelFileName, ListFile: string;
  MinSymbols: Integer = 50;                 // Minimum for loading.
  MinTokens: Integer = 50;                  // Minimum for loading.
  MinCorpus: Integer = 50;                  // Minimum for loading.
  // Utility vars.
  Ch: string;                               // For option menu.
  CombinedSymbolTable: TSymbolTable;        // For combining two symbol tables.
  i: Integer;

{  Writeln('K: Tokenize -- Create a token list from a corpus. Output is a token list. A symbol list');
  Writeln('(which may include a merge list) may be created from the corpus or input by the user.');
  Writeln('Required is a corpus, and, if desired, a symbol list list.');
  Writeln('T: Train -- Train a model (that is, find parameters) on a token list.');
  Writeln('Required is token list.');
  Writeln('I: Infer -- Run a model forward doing inference. Required is a model,');
  Writeln('a token list, a symbol list, and a query.');
  Writeln('U: Utilities -- Combine two symbol lists.');
  Writeln('B: Use Bela corpus. DT: Use Damned Thing token list.');}



  // Create and name directory and file for saving.
Procedure LogFile(const Eponym: string);
var
  SaveOut: Text;                            // Save Output mode.
begin
  WorkingDir := ChangeFileExt(Eponym, '') + FormatDateTime('yyyy-mm-dd_hhnnss', Now);
  WorkingName := WorkingDir;                // Working directory.
  CreateDir(WorkingDir);                    // Create folder of files.
  ChDir(WorkingDir);                        // And go there.

  // Save current Output.
  SaveOut := Output;

  // Redirect Output.
  Assign(Output, WorkingName + '.log');     // Create log file in folder.
  ReWrite(Output);
  ReportInfo;                               // Write report of info in folder.

  // Restore Output to console.
  Close(Output);
  Output := SaveOut;                        // Go back to console.
  ChDir('..');                              // Go back to parent directory.
 end;

// Read a file of file names, and sends each to tokenizer.
procedure ProcessFileList(var ListFile: string; var Corpus: TBVector);
var
  F: TextFile;               // ListFile is the file of corpus file names.
  Line: string;              // Line is one corpus file name.
  FilesRead: TSVector;       // List of file names read.
  OneCorpus: TBVector;       // One corpus to concatenate.
  Count: Integer;
begin
  MultipleFileName := EmptyStr;        // This var contains info on input corpuses.
  Write('Enter name of file list: ');
  Readln(ListFile);
  if not FileExists(ListFile) then begin
    Writeln('List file not found: ', ListFile);
    Exit;
  end;

  AssignFile(F, ListFile);
  Reset(F);

  Count := 0;                          // Count the input corpuses.
  SetLength(FilesRead, 0);
  FromSymbolTable := False;            // Tells whether there's a symboltable.
  SetLength(Corpus, 0);                // Replace with length(ST)?

  while not EOF(F) do begin            // Loop thru the corpuses.
    ReadLn(F, Line);
    Line := Trim(Line);
    if Line = '' then Continue;         // Skip blank lines.
    if not FileExists(Line) then begin
      Writeln('  File not found: ', Line, '.');
      Continue;
    end;
    if (Count = 0) and SaveFiles then
      LogFile('Mult' + ListFile);

    ReadFileBytes(Line, OneCorpus);     // Read the file into OneCorpus.
    SetLength(CorpusFileNames, Count + 1);
    CorpusFileNames[Count] := Line;
    Writeln('  File processed: ', Line, '; corpus bytes read: ', Length(OneCorpus), '.');
    if Length(OneCorpus) < MinCorpus then begin
      Writeln('Corpus too small. Aborting...');
      Continue;
    end;

    Corpus := Concat(Corpus, OneCorpus);     // Concat Corpus with OneCorpus.
    nCorpus := Length(Corpus);
    Writeln('Total bytes read: ', Length(Corpus), '.');
    Inc(Count);
    SetLength(FilesRead, Count);
    FilesRead[Count - 1] := Line;
  end;

  CloseFile(F);

  Writeln('Combined corpus length = ', Length(Corpus));
  nCorpus := Length(Corpus);
  Pause;
end;

// Options file.
procedure Options;
begin
  Writeln('Options:');
  Writeln('  1: Tokenize an input corpus from a file using WesChat''s byte-level byte-pair encoding, with');
  Writeln('     deterministic left-to-right longest-prefix matching and greedy longest-match decoding.');
  Writeln('  2: Tokenize an input set of corpuses listed one per line in a file, to create a concatenated token list,');
  Writeln('     using WesChat''s tokenization routine.');
  Writeln('  3: Tokenize bela corpus using WesChat''s tokenization routine.');
  Writeln('  4: Tokenize an input corpus, based on an input symbol table, using WesChat''s tokenization routine.');
  Writeln('  5: Tokenize an input corpus using ChatGPT''s symbol and merge tables and WesChat''s');
  Writeln('     tokenization routine.');
  Writeln('  6: Input a token list to be used in training.');
  Writeln('  7: Combine two symbol tables.');
  Writeln('  8: Tokenize an input set of corpuses listed one per line in a file, to create a concatenated token list,');
  Writeln('     using an input symbol table and WesChat''s tokenization routine.');
  Writeln('  9: Create symbol table from input corpus.');
  Writeln('  10: Save a model.');
  Writeln('  11: Load a model, load a token list, and run training or inference.');
  Writeln('  13: Load token list and symbol table for dt327.');
  Writeln('  H: Help.');
  Writeln('  X: Exit.');
end;

// Help file.
procedure Help;
begin
  Options;
  Writeln('  VTO: VerboseTokenize := True                  NVTO: VerboseTokenize := False');
  Writeln('  DC: DisplayCorpus := True                     NDC: DisplayCorpus := False');
  Writeln('  DTW: DisplayTokenWork := True                 NDTW: DisplayTokenWork := False');
  Writeln('  DMW: DisplayMergeWork := True                 NDMW: DisplayMergeWork := False');
  Writeln('  DV: DisplayVerification := True               NDV: DisplayVerification := False');
  Writeln('  DEBR: DisplayEachByteRead := True             NDEBR: DisplayEachByteRead := False');
  Writeln('  SPST: SavePartialSymbolTable := True          NSPST: SavePartialSymbolTable := False');
  Writeln('  DW: DisplayWindow := True                     NDW: DisplayWindow := False');
  Writeln('  DNP: DoNotPause := True                       DP: DoNotPause := False');
  Writeln('  SF: SaveFiles := True                         NSF: SaveFiles := False');

  Writeln('  M: Maximum merges: ');
  Writeln('  PC: Maximum pair count: ');
  Writeln('  PSTT: PartialSymbolTableTrigger: ');
  Writeln('  LR: Override learning rate: ');
  Writeln('  T: Temperature: ');

  Writeln('  DE: DisplayEpoch := True; DisplayStage := False; DisplaySubstage := False');
  Writeln('  DS: DisplayStage := True; DisplayEpoch := True; DisplaySubstage := False');
  Writeln('  DSS: DisplaySubstage := True; DisplayStage := True; DisplayEpoch := True');
  Writeln('  ND: DisplayEpoch := False; DisplayStage := False; DisplaySubsubstage := True');
end;

// Helper function for proceeding to Train.
function QueryTrain: Boolean;
begin
  Write('Do you wish to proceed to training? (y/n) ');
  Readln(Ch);
  if UpCase(Ch) = 'N' then
    Result := False
  else
    Result := True;
end;

// Helper function for proceeding to Infer.
function QueryInfer: Boolean;
begin
  Write('Do you wish to proceed to inference? (y/n) ');
  Readln(Ch);
  if UpCase(Ch) = 'N' then
    Result := False
  else
    Result := True;
end;

// Start of main program.
begin
  { Necessary because JSON will throw dupe errors otherwise }
  SetMultiByteConversionCodePage(CP_UTF8);
  SetMultiByteRTLFileSystemCodePage(CP_UTF8);

  // More startup for GPT2.
  Vocab := TStringList.Create;
  LoadVocab('vocab.json', Vocab);

  { Below is not working on my Lazarus console }
  SetConsoleOutputCP(CP_UTF8);
  SetConsoleCP(CP_UTF8);

  { Possible CLI -- need model, model = tokenlist + params, tokenlist = wes symbol table or gpt symbol and merge tables }
  // tokenize: westok, gpttok + filenames (1 or more)
  // run one preset: bela winston
  // input symbol table: symtab + filename
  // input merge table: mergetab + filename
  // input token list: toklist + filename
  // combine 2 symbol tables: comb (enhance to 2+)
  // create wes symbol table: symb + filename
  // load model: load, save model: save + filenames
  // perform forward inference on model: infer + optional filenames

  Writeln('WesChat, Version 1.2, begun January 19, 2026, by Wesley R. Parsons, wespar@bellouth.net, www.wesparsons.com.');
  Writeln;
  Options;
  Writeln;
  Writeln('The symbol table and other information, including if desired the token list, will be written to disk.');
  Writeln('After tokenization, WesChat prompts for training the transformer. It consists');
  Writeln('of multiple blocks (default is 4). The attention stage has multiple heads (defualt is 0). There are weight stages');
  Writeln('with and without bias. The activation function is softmax with temperature.');
  Writeln('Default model dimensions are 512. The activation stage expands dimensionality fourfold.');
  Writeln('Precision is single. Default sequence length is 128 or 256 bytes. Pre-layer normalization');
  Writeln('standardizes for means and standard deviations. Deafult sttention, MLP, and residual dropouts are 0.1.');
  Writeln('The softmax function normalizes exponentially with a default temperature of 1.0. The learning rate may be set.');
  Writeln('All output files will be contained in a folder or file named with the input file name,');
  Writeln('appended with a timestamp. After training, a model will run forward inference.');
  while True do begin
    Write('W>');
    Readln(Ch);
    Case UpperCase(Ch) of
      '1': begin
        FromSymbolTable := False;
        nSymbols := 0;
        SetLength(SymbolTable, 0);
        SetLength(TokenizedCorpus, 0);
        MaxMerges := 20000;
        MaxPairCount := 400000;

        // Ask user for corpus file.
        Write('Enter corpus file name: ');
        Readln(CorpusFileName);

        // Check existence and size of corpus file .
        if not FileExists(CorpusFileName) then begin
          Writeln('  File not found: ', CorpusFileName, '.');
          Continue;
        end;
        if FileSize(CorpusFileName) < MinCorpus then begin
          Writeln('Corpus too small. Aborting...');
          Continue;
        end;

        // Read corpus bytes from file.
        ReadFileBytes(CorpusFileName, Corpus);
        nCorpus := Length(Corpus);
        SetLength(CorpusFileNames, 1);
        CorpusFileNames[0] := CorpusFileName + '   ' + IntToStr(FileSize(CorpusFileName))
         + ' bytes   ' + DateTimeToStr(FileDateToDateTime(FileAge(CorpusFileName)));

        // Write to log file.
        if SaveFiles then
          LogFile(CorpusFileName);

        if MaxMerges <= 0 then begin
          Writeln('MaxMerges was ', MaxMerges, '; resetting to 20000.');
          MaxMerges := 20000;
        end;

        if MaxPairCount <= 0 then begin
          Writeln('MaxPairCount was ', MaxPairCount, '; resetting to 400000.');
          MaxPairCount := 400000;
        end;

        // Run WesChat symbolizer.
        RunSymbolize(Corpus);

        Writeln('After RunSymbolize: nSymbols=', nSymbols,
                ' Length(SymbolTable)=', Length(SymbolTable),
                ' MaxMerges=', MaxMerges,
                ' MaxPairCount=', MaxPairCount,
                ' DimVocab=', DimVocab);

        if Length(SymbolTable) < MinSymbols then begin
          Writeln('Too few symbols found after RunSymbolize. Aborting...');
          Pause;
          Continue;
        end;

        Tokenizer := WesTokenizer;
        RunWesTokenize(Corpus, TokenizedCorpus);
        PadToSeqMultiple(TokenizedCorpus, SeqLen);
        // Run tokenizer.
        Tokenizer := WesTokenizer;
        RunWesTokenize(Corpus, TokenizedCorpus);
        PadToSeqMultiple(TokenizedCorpus, SeqLen);

        // Check number of symbols.
        if nSymbols < MinSymbols then begin
          Writeln('Too few symbols found. Aborting...');
          Continue;
        end;

        // Train.
        if QueryTrain then begin
          RunTrain(WModelParams, WModelState, TokenizedCorpus);
          if QueryInfer and TrainSuccess then
            RunInfer(WModelParams, WModelState);
        end;
      end;
      '2': begin

        FromSymbolTable := False;
        nSymbols := 0;
        SetLength(SymbolTable, 0);
        SetLength(TokenizedCorpus, 0);

        // Process multiple corpuses.
        ProcessFileList(ListFile, Corpus);

        // Run WesChat symbolizer.
        RunSymbolize(Corpus);

        // Display symboltable.
        DisplayByteSymbolTable(SymbolTable);

        // Run tokenizer.
        Tokenizer := WesTokenizer;
        RunWesTokenize(Corpus, TokenizedCorpus);
        PadToSeqMultiple(TokenizedCorpus, SeqLen);

        // Train.
        If QueryTrain then begin
          RunTrain(WModelParams, WModelState, TokenizedCorpus);
          if QueryInfer and TrainSuccess then
            RunInfer(WModelParams, WModelState);
        end;
      end;
      '3': begin
        // Read corpus file.
        ReadFileBytes('bela.txt', Corpus);
        FromSymbolTable := True;
        nCorpus := Length(Corpus);
        FileName := 'bela.txt';

        // Read symbol table file.
        SymbolFileName := 'bela.sym';
        SetLength(CorpusFileNames, 1);
        CorpusFileNames[0] := SymbolFileName;

        // Read symboltable.
        LoadSymbolTable(SymbolFileName, SymbolTable);

        // Write to log file.
        if SaveFiles then
          LogFile('bela.txt');

        // Run tokenizer.
        Tokenizer := WesTokenizer;
        RunWesTokenize(Corpus, TokenizedCorpus);
        PadToSeqMultiple(TokenizedCorpus, SeqLen);

        // Run Train.
        if QueryTrain then begin
          RunTrain(WModelParams, WModelState, TokenizedCorpus);
          if QueryInfer and TrainSuccess then
            RunInfer(WModelParams, WModelState);
        end;
      end;
      '4': begin
        // Ask user for corpus file.
        Write('Input corpus file name: ');
        Readln(CorpusFileName);

        // Check existence and size of corpus file.
        if not FileExists(CorpusFileName) then begin
          Writeln('File not found: ', CorpusFileName, '. Aborting...');
          Continue;
        end;

        // Ask user for symbol file.
        Write('Input symbol table file name: ');
        Readln(SymbolFileName);
        FromSymbolTable := True;  // Do I need this var. Length(ST) = 0.

        // Check existence of symbol file.
        if not FileExists(SymbolFileName) then begin
          Writeln('File not found: ', SymbolFileName, '. Aborting...');
          Continue;
        end;

        // Read the symbol table.
        LoadSymbolTable(SymbolFileName, SymbolTable);

        // Check size of symbol table.
        if Length(SymbolTable) < MinSymbols then begin
          Writeln('Too few symbols found. Aborting...');
          Continue;
        end;

        // Read corpus bytes from file.
        ReadFileBytes(CorpusFileName, Corpus);
        nCorpus := Length(Corpus);

        if Length(Corpus) < MinCorpus then begin
          Writeln('Corpus too small. Aborting...');
          Continue;
        end;

        SetLength(CorpusFileNames, 1);
        CorpusFileNames[0] := CorpusFileName;

        // Write to log file.
        if SaveFiles then
          LogFile(CorpusFileName);

        // Run tokenizer.
        Tokenizer := WesTokenizer;
        RunWesTokenize(Corpus, TokenizedCorpus);
        PadToSeqMultiple(TokenizedCorpus, SeqLen);

        // Run Train.
        if QueryTrain then begin
          RunTrain(WModelParams, WModelState, TokenizedCorpus);
          if QueryInfer and TrainSuccess then
            RunInfer(WModelParams, WModelState);
        end;
      end;
      '5': begin
        // Ask user for corpus file.
        Write('Enter corpus file name: ');
        Readln(CorpusFileName);

        // Check corpus file for existence and size.
        if not FileExists(CorpusFileName) then begin
          Writeln('File not found: ', CorpusFileName, '.');
          Continue;
        end;
        if FileSize(CorpusFileName) < MinCorpus then begin
          Writeln('Corpus too small. Aborting...');
          Continue;
        end;

        // Read bytes from file.
        ReadFileBytes(CorpusFileName, Corpus);
        FromSymbolTable := True;
        nCorpus := Length(Corpus);
        SetLength(CorpusFileNames, 1);
        CorpusFileNames[0] := CorpusFileName;

        // Write to log file.
        if SaveFiles then
          LogFile(CorpusFileName);

        // Run tokenizer.
        Tokenizer := GPT2Tokenizer;
        RunGPT2Tokenize(CorpusFileName, TokenizedCorpus);
        PadToSeqMultiple(TokenizedCorpus, SeqLen);

        // Check tokenized corpus.
        Writeln('First 200 token of tokenized corpus: ');
        for i := 0 to Min(199, High(TokenizedCorpus)) do
          Write(TokenizedCorpus[i], ' ');
        Writeln;
        Pause;

        // Check number of symbols, and Train.
        if nSymbols > 0 then begin
          RunTrain(WModelParams, WModelState, TokenizedCorpus);
          if QueryInfer and TrainSuccess then
            RunInfer(WModelParams, WModelState);
        end
        else
          Writeln('Symbols not found in table.');
      end;
      '6': begin
        // Ask user for token file.
        Write('Enter token list file name: ');
        Readln(TokenFileName);

        // Check existence and size of token file.
        if not FileExists(TokenFileName) then begin
          Writeln('File not found: ', TokenFileName, '.');
          Continue;
        end;

        // Read token file.
        IOHandler.LoadTokenList(TokenFileName, TokenizedCorpus);

        // Check size of token file.
        if Length(TokenizedCorpus) < MinTokens then begin
          Writeln('Token list too small. Aborting...');
          Continue;
        end
        else
          PadToSeqMultiple(TokenizedCorpus, SeqLen);

        // Ask user for symbol file.
        Write('Input symbol table file name: ');
        Readln(SymbolFileName);
        FromSymbolTable := True;  // Do I need this var. Length(ST) = 0.

        // Check existence of symbol file.
        if not FileExists(SymbolFileName) then begin
          Writeln('File not found: ', SymbolFileName, '. Aborting...');
          Continue;
        end;

        // Read the symbol table.
        LoadSymbolTable(SymbolFileName, SymbolTable);

        // Check size of symbol table.
        if Length(SymbolTable) < MinSymbols then begin
          Writeln('Too few symbols found. Aborting...');
          Continue;
        end;

        // Display full corpus, tokenized and detokenized.
        // TCFull(TokenizedCorpus);

        // Run Train.
        If QueryTrain then begin
          RunTrain(WModelParams, WModelState, TokenizedCorpus);
          if QueryInfer and TrainSuccess then
            RunInfer(WModelParams, WModelState);
        end;
      end;
      '7': begin
        // Merge symbol tables.
        MergeSymbolTables(CombinedSymbolTable);

        // Ask user for output symbol table name.
        Write('Output symbol table name:');
        Readln(SymbolFileName);

        // Write to log file.
        if SaveFiles then
          LogFile(SymbolFileName);

        // Save combined symboltable.
        SaveSymbolTable(SymbolFileName, CombinedSymbolTable);
        Writeln('File ', SymbolFileName, ' successfully saved.');
        Writeln;
      end;
      '8': begin
        // Process multiple corpuses.
        ProcessFileList(ListFile, Corpus);

        // Ask user for symbol table file.
        Write('Input symbol table file name: ');
        Readln(SymbolFileName);
        FromSymbolTable := True;

        // Check for existence of symboltable.
        if not FileExists(SymbolFileName) then begin
          Writeln('Symbol table file not found: ', SymbolFileName, '. Aborting...');
          Continue;
        end;

        // Read symboltable.
        LoadSymbolTable(SymbolFileName, SymbolTable);

        // Check size of symboltable.
        if Length(SymbolTable) < MinSymbols then begin
          Writeln('Too few symbols found. Aborting...');
          Continue;
        end;

        // Display symboltable.
        DisplayByteSymbolTable(SymbolTable);

        // Run tokenizer.
        Tokenizer := WesTokenizer;
        RunWesTokenize(Corpus, TokenizedCorpus);
        PadToSeqMultiple(TokenizedCorpus, SeqLen);

        // RunTrain.
        If QueryTrain then begin
          RunTrain(WModelParams, WModelState, TokenizedCorpus);
          if QueryInfer and TrainSuccess then
            RunInfer(WModelParams, WModelState);
        end;
      end;
      '9': begin
        // Ask user for corpus file.
        Write('Enter corpus file name: ');
        Readln(CorpusFileName);

        // Check existence and size of corpus file.
        if not FileExists(CorpusFileName) then begin
          Writeln('File not found: ', CorpusFileName, '.');
          Continue;
        end;
        if FileSize(CorpusFileName) < MinCorpus then begin
          Writeln('Corpus too small. Aborting...');
          Continue;
        end;

        // Read bytes from file.
        ReadFileBytes(CorpusFileName, Corpus);
        nCorpus := Length(Corpus);
        SetLength(CorpusFileNames, 1);
        CorpusFileNames[0] := CorpusFileName + '   ' + IntToStr(FileSize(CorpusFileName))
          + ' bytes   ' + DateTimeToStr(FileDateToDateTime(FileAge(CorpusFileName)));

        // Write to Log file.
        if SaveFiles then
          LogFile(CorpusFileName);

        // Run WesChat symbolizer.
        RunSymbolize(Corpus);

        // Display symbol table.
        DisplayByteSymbolTable(SymbolTable);
      end;
      '10': begin       // Save model.
        Write('Enter filename: ');
        Readln(ModelFileName);
        if SaveModel(ModelFileName, WModelParams) then
           Writeln('File ', ModelFileName, ' successfully saved.')
        else
          Writeln('File not saved.');
        Pause;
      end;
      '11': begin       // Load model, and then tokenized corpus.
        Write('Enter filename: ');
        Readln(ModelFileName);

        // Check existence and size of file.
        if not FileExists(ModelFileName) then begin
          Writeln('  File not found: ', ModelFileName, '.');
          Continue;
        end;

        if CudaAllocated then
          MDeallocateCublas(WModelParams, WModelState);

        if LoadModel(ModelFileName, WModelParams) then begin
          Writeln('File ', ModelFileName, ' loaded.');
        end                     // Need to do   StartCuda(WModelParams, WModelState);
        else
          Writeln('File not loaded.');
        Pause;

        // Ask user for token file.
        Write('Enter token list file name: ');
        Readln(TokenFileName);

        // Check existence and size of token file.
        if not FileExists(TokenFileName) then begin
          Writeln('File not found: ', TokenFileName, '.');
          Continue;
        end;

        // Read token file.
        IOHandler.LoadTokenList(TokenFileName, TokenizedCorpus);

        // Check size of token file.
        if Length(TokenizedCorpus) < MinTokens then begin
          Writeln('Token list too small. Aborting...');
          Continue;
        end
        else
          PadToSeqMultiple(TokenizedCorpus, SeqLen);

        // Ask user for symbol file.
        Write('Input symbol table file name: ');
        Readln(SymbolFileName);
        FromSymbolTable := True;  // Do I need this var. Length(ST) = 0.

        // Check existence of symbol file.
        if not FileExists(SymbolFileName) then begin
          Writeln('File not found: ', SymbolFileName, '. Aborting...');
          Continue;
        end;

        // Read the symbol table.
        LoadSymbolTable(SymbolFileName, SymbolTable);

        // Check size of symbol table.
        if Length(SymbolTable) < MinSymbols then begin
          Writeln('Too few symbols found. Aborting...');
          Continue;
        end;

        Write('Run Training or Inference? (Enter T or I): ');
        Readln(Ch);

        if Ch ='T' then begin
          // RunTrain.
          RunTrain(WModelParams, WModelState, TokenizedCorpus);
          if QueryInfer and TrainSuccess then
            RunInfer(WModelParams, WModelState);
        end
        else begin
          ParamsNeedCopyToDevice := True;
          RunInfer(WModelParams, WModelState);
        end;
      end;
      '13': begin
        // Ask user for token file.
        TokenFileName := 'dt327.tok';

        // Read token file.
        IOHandler.LoadTokenList(TokenFileName, TokenizedCorpus);

        PadToSeqMultiple(TokenizedCorpus, SeqLen);

        SymbolFileName := 'dt327.sym';
        FromSymbolTable := True;  // Do I need this var. Length(ST) = 0.

        // Read the symbol table.
        LoadSymbolTable(SymbolFileName, SymbolTable);

        // Display full corpus, tokenized and detokenized.
        // TCFull(TokenizedCorpus);

        // Run Train.
        If QueryTrain then begin
          RunTrain(WModelParams, WModelState, TokenizedCorpus);
          if QueryInfer and TrainSuccess then
            RunInfer(WModelParams, WModelState);
        end;
      end;
      'X':     Exit;
      'H':     Help;

      'VTO':   VerboseTokenize := True;
      'NVTO':  VerboseTokenize := False;
      'DC':    DisplayCorpus := True;
      'NDC':   DisplayCorpus := False;
      'DTW':   DisplayTokenWork := True;
      'NDTW':  DisplayTokenWork := False;
      'DMW':   DisplayMergeWork := True;
      'NDMW':  DisplayMergeWork := False;
      'DV':    DisplayVerification := True;
      'NDV':   DisplayVerification := False;
      'DEBR':  DisplayEachByteRead := True;
      'NDEBR': DisplayEachByteRead := False;
      'SPST':  SavePartialSymbolTable := True;
      'NSPST': SavePartialSymbolTable := False;

      'VTR':   VerboseTransform := True;
      'NVTR':  VerboseTransform := False;
      'DE':    begin DisplayEpoch := True; DisplayStage := False; DisplaySubstage := False; end;
      'DS':    begin DisplayStage := True; DisplayEpoch := True; DisplaySubstage := False; end;
      'DSS':   begin DisplaySubstage := True; DisplayStage := True; DisplayEpoch := True; end;
      'ND':    begin DisplayEpoch := False; DisplayStage := False; DisplaySubstage := True; end;
      'DW':    DisplayWindow := True;
      'NDW':   DisplayWindow := False;

      'DNP':   DoNotPause := True;
      'DP':    DoNotPause := False;

      'SF':    SaveFiles := True;
      'NSF':   SaveFiles := False;

      'T':     begin
        Write('Temperature: ');
        Readln(Temperature);
      end;
      'LR':    begin
        Write('Override learning rate: ');
        Readln(OverrideLearningRate);
      end;
      'M': begin
        Write('Maximum merges: ');
        Readln(MaxMerges);
      end;
      'PC': begin
        Write('Maximum pair count: ');
        Readln(MaxPairCount);
      end;
      else Writeln('Invalid input');
//      'PSTT':  PartialSymbolTableTrigger: ');
    end;
  end;

  // Clean up cublas.
  EndCuda(WModelParams, WModelState);
  {if CudaAllocated then
    MDeallocateCublas(WModelParams, WModelState);
  if CublasInitialized then
    cublasDestroy_v2(CuHandle);}

  // Free Vocab.
  Vocab.Free;
end.}
