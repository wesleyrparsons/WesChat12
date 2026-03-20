unit WesTokenize;

{$mode ObjFPC}{$H+}{$I proprietary.txt}

{ WesChat, Version 1.2, begun January 10, 2026, by Wesley R. Parsons, wespar@bellouth.net, www.wespar.com.}

interface

uses
  Classes,
  Crt,
  DateUtils,
  Display,
  FileUtil,
  Global,
  SysUtils;

type
  TSymbolTable = TRBSVector;           // Array of symbols. So index of array is a symbol string.
  TTokenCount = record                 // Records count of tokens.
    Symbol: Integer;                   // Symbol for token.
    Count: Integer;                    // Number of times it occurs.
  end;
  TTokenCounts = array of TTokenCount; // Array of token counts.

  PTrieNode = ^TTrieNode;
  TTrieNode = record
    Children: array[0..255] of PTrieNode; // ASCII.
    TokenID: Integer;                     // -1 if not terminal.
  end;

// Token statistics.
  TMergedTokenStat = record
    TokenID: Integer;
    Count: Integer;
  end;

  TMergedTokenStats = array of TMergedTokenStat;
var
  StartSymbol: Integer = 260;                    // UTF-8 0.255, BOS, EOS, PAD, UNK is 259.
  TokenizedCorpus: TIVector;
  SymbolTable: TSymbolTable;
  BOS, EOS, PAD, UNK: Integer;                   // Extra symbols for control.
  ElapsedMS: Int64;                              // For timing.
  Hours, Mins: Int64;                            // For timing.
  Secs, MSecs: Double;                           // For timing.
  FileName, WorkingName, Stamp,
    Reconstructed: String;                       // Saving data.
  Magic: array[0..3] of Char = ('S', 'Y', 'M', 'T');  // For saving symbol table.
  TrieHead: PTrieNode = nil;                     // Nodes for Trie.
  MergedTypes, UnmergedTypes: Integer;
  MergedInstances, UnmergedInstances: Integer;
  i: Integer;

procedure ReadFileBytes(const FileName: String; var OneCorpus: TBVector);
procedure WriteTokenList(const Part: TPart = B);
procedure LoadTokenList(const BinFileName: String);
procedure LoadSymbolTable(const FileName: string; var SymbolTable: TSymbolTable);
procedure DisplaySymbolTable;
procedure TokenizeFromSymbolTable(const TextFileName: string; var Corpus: TBVector);
procedure BuildTrie(const SymbolTable: TRBSVector; out Root: PTrieNode);
function MatchLongest(root: PTrieNode; const text: TBVector; startPos: Integer;
  out tokenID, matchLen: Integer): Boolean;
procedure DetokenizeToDisplay(const TokenizedCorpus: TIVector; const Part: TPart = B);
procedure ReportStatistics;
procedure RunWesTokenize(var Corpus: TBVector; const SymbolTable: TSymbolTable);

implementation

{ Load the Corpus }
// Read the corpus as a stream of binary.
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
  if VeryVerbose and VerboseTokenize then
    writeln('--- Original Corpus ---');
  for i := 0 to Size - 1 do begin
    BlockRead(F, B, 1);
    OneCorpus[i] := B;

    if VeryVerbose and VerboseTokenize then
      if ShowEachByteRead then
        if B < 32 then
          Write('<', B, '>')
        else
          Write(Chr(B));
  end;
  CloseFile(F);
  if VeryVerbose and VerboseTokenize then begin
    writeln('ReadByteFile: ');
    for i := 0 to 150 do
      write(OneCorpus[i], ' ');
    Pause;
  end;
  if VeryVerbose and VerboseTokenize then
    writeln;

  // Display initial Corpus length.
  Writeln('Read ', Size, ' bytes from ', FileName);
end;

{ Construct the token linked list }
{ PIPELINE 2: Use existing symbol table }
// Load the symbol table from file.
procedure LoadSymbolTable(const FileName: string; var SymbolTable: TSymbolTable);
var
  F: file;
  Magic: array[0..3] of Char;
  i, Len: Integer;
  S: string;

begin
  Assign(F, FileName);
  Reset(F, 1);

  // Magic header.
  BlockRead(F, Magic, SizeOf(Magic));
  if (Magic[0] <> 'S') or (Magic[1] <> 'Y') or
     (Magic[2] <> 'M') or (Magic[3] <> 'T') then begin
    Close(F);
    writeln('Invalid symbol table file.');
    Pause;
    Exit;
  end;

  // Version.
  BlockRead(F, Version, 16);

  // Symbol count.
  BlockRead(F, nSymbols, SizeOf(nSymbols));
  SetLength(SymbolTable, NSymbols);

  // Special token IDs
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
  nVocab := nSymbols;
  Writeln('Loaded ', nSymbols, ' symbols from ', FileName);
end;

// Trie procedures.
// Call Once.
procedure InsertTrieSymbol(root: PTrieNode; const s: string; id: Integer);
var
  node: PTrieNode;
  i: Integer;
  c: Byte;
begin
  node := root;

  for i := 1 to Length(s) do begin
    c := Ord(s[i]);

    if node^.Children[c] = nil then begin
      New(node^.Children[c]);
      FillChar(node^.Children[c]^, SizeOf(TTrieNode), 0);
      node^.Children[c]^.TokenID := -1;
    end;

    node := node^.Children[c];
  end;

  node^.TokenID := id;  // mark terminal
end;

// Build trie.
procedure BuildTrie(const SymbolTable: TRBSVector; out Root: PTrieNode);
var
  i: Integer;
begin
  New(Root);
  FillChar(Root^, SizeOf(TTrieNode), 0);
  Root^.TokenID := -1;

  for i := 0 to High(SymbolTable) do begin
    // Skip special tokens 256..259.
    if (i = BOS) or (i = EOS) or (i = PAD) or (i = UNK) then Continue;
    if SymbolTable[i] = '' then Continue;

    InsertTrieSymbol(Root, SymbolTable[i], i);
  end;
end;

function MatchLongest(root: PTrieNode;
  const text: TBVector;
  startPos: Integer;
  out tokenID,
  matchLen: Integer): Boolean;
var
  node: PTrieNode;
  i: Integer;
  c: Byte;
  lastMatchID: Integer;
  lastMatchLen: Integer;
begin
  node := root;
  lastMatchID := -1;
  lastMatchLen := 0;

  i := startPos;

  while (i < Length(text)) do  begin
    c := text[i];

    if node^.Children[c] = nil then break;

    node := node^.Children[c];

    if node^.TokenID <> -1 then begin
      lastMatchID := node^.TokenID;
      lastMatchLen := i - startPos + 1;
    end;

    Inc(i);
  end;

  if lastMatchID <> -1 then begin
    tokenID := lastMatchID;
    matchLen := lastMatchLen;
    Result := True;
  end
  else
    Result := False;
end;

// Tokenize Corpus from SymbolTable loaded by program.
procedure TokenizeFromSymbolTable(const TextFileName: string; var Corpus: TBVector);
var
  i, BestSym, BestLen: Integer;
begin
  if FileExists(TextFileName) then
    ReadFileBytes(TextFileName, Corpus);

  nCorpus := Length(Corpus);
  SetLength(TokenizedCorpus, 0);

  i := 0;

  BuildTrie(SymbolTable, TrieHead);

  while i < nCorpus do begin
    if MatchLongest(TrieHead, Corpus, i, BestSym, BestLen) then begin
      SetLength(TokenizedCorpus, Length(TokenizedCorpus) + 1);
      TokenizedCorpus[High(TokenizedCorpus)] := BestSym;
      Inc(i, BestLen);
    end
    else begin
      // Fallback: single byte token.
      SetLength(TokenizedCorpus, Length(TokenizedCorpus) + 1);
      TokenizedCorpus[High(TokenizedCorpus)] := Corpus[i];
      Inc(i);
    end;
  end;

  nTokenizedCorpus := Length(TokenizedCorpus);

  writeln('Created ', nTokenizedCorpus, ' tokens from ', TextFileName);

  if VerboseTokenize then Begin
    writeln('First 50 token of tokenized corpus');
    for i := 0 to 49 do
      write(TokenizedCorpus[i], ' ');
    writeln;
    Pause;
  end;

  nVocab := nSymbols;
  writeln('End of tokenization. Press <CR> to continue.');
end;

{ Create the tokenized corpus from linked list }
{ Display routines }
// Display the symbol table.
procedure DisplaySymbolTable;
var
  i, j: Integer;
  Disp: String;
begin
  Writeln('--- Symbol Table ---');           // Chr(183) is non-display char.
  for i := 0 to High(SymbolTable) do         // Loop thru each symbol in table.
    if SymbolTable[i] <> '' then begin
      Disp := SymbolTable[i];                // Use Disp so Table is not changed.
      for j := 1 to Length(SymbolTable[i]) do
        if (Ord(Disp[j]) < 32) or (Ord(Disp[j]) = 127) then Disp[j] := Chr(183);
      if Length(Disp) < 12 then begin
        write(i: 8, Disp: 15);
        if (i mod 5) = 4 then writeln;
      end
      else begin
        if not (i mod 5) = 4 then writeln;
        Writeln(i: 8, '     ', Disp);
      end;
      if (i > 0) and (i mod 100 = 99) then Pause;
    end;
  writeln('Symbol table length = ', Length(SymbolTable));
  writeln;
end;

{ Reconstruction check }
{ Computations and reports }
// Calculate time statistics.
procedure CalculateTimeStatistics;
begin
  // Total elapsed time.
  ElapsedMS := MilliSecondsBetween(t0, t1) - Round(StopTime);
  Hours := ElapsedMS div 3600000;
  Mins := ElapsedMS div 60000;
  Secs := (ElapsedMS mod 60000) / 1000.0;
end;

// Calculate number of symbol types and instances.
procedure CalculateSymbolCount;
var
  i, T: Integer;
begin
  MergedTypes := 0;
  UnmergedTypes := 0;

  // Count symbol types.
  for i := 0 to High(SymbolTable) do
    if Length(SymbolTable[i]) > 1 then
      Inc(MergedTypes)
    else
      Inc(UnmergedTypes);

  // Count token instances.
  MergedInstances := 0;
  UnmergedInstances := 0;

  for i := 0 to High(TokenizedCorpus) do begin
    T := TokenizedCorpus[i];
    if Length(SymbolTable[T]) > 1 then
      Inc(MergedInstances)
    else
      Inc(UnmergedInstances);
  end;
end;

// Count the number of occurrences of each symbol.
procedure CountSymbols(const SymbolTable: TSymbolTable);
var
  Counts, Index: TIVector;
  i, j, k, N, TmpIndex: Integer;
begin
  // Allocate and zero Counts.
  SetLength(Counts, Length(SymbolTable));
  FillChar(Counts[0], SizeOf(Counts[0]), 0);

  // Count occurrences.
  for i := 0 to High(TokenizedCorpus) do
    Inc(Counts[TokenizedCorpus[i]]);

  // Build index array.
  N := Length(Counts);
  SetLength(Index, N);
  for i := 0 to N - 1 do
    Index[i] := i;

  // Selection sort index array by Counts[index] descending.
  for i := 0 to N - 2 do begin
    k := i;
    for j := i + 1 to N-1 do
      if Counts[Index[j]] > Counts[Index[k]] then
        k := j;

    // Swap Index[i] and Index[k].
    TmpIndex := Index[i];
    Index[i] := Index[k];
    Index[k] := TmpIndex;
  end;

  // Print top 60.
  writeln('Top 60 most frequent symbols:');
  for i := 0 to 59 do begin
    k := Index[i];
    write(i + 1: 8, ': Symbol ', k: 8, '  Count=', Counts[k]: 6, '  ', '"' + SymbolTable[k] + '"': 15);
    if ((i + 1) mod 3) = 0 then
      writeln;
  end;

  Pause;
end;

// Calculate token statistics.
// Count token usage.
procedure CountTokenUsage(const TokenizedCorpus: TIVector; nSymbols: Integer; var Counts: TIVector);
var
  i, t: Integer;
begin
  SetLength(Counts, nSymbols);

  for i := 0 to nSymbols - 1 do
    Counts[i] := 0;

  for i := 0 to High(TokenizedCorpus) do begin
    t := TokenizedCorpus[i];
    if (t >= 0) and (t < nSymbols) then
      Inc(Counts[t]);
  end;
end;

// Build a list of merged token states.
procedure BuildMergedTokenStats(const Counts: TIVector; FirstMergedToken: Integer;
  out Stats: TMergedTokenStats);
var
  i, k: Integer;
begin
  SetLength(Stats, 0);
  k := 0;

  for i := FirstMergedToken to High(Counts) do begin
    SetLength(Stats, k + 1);
    Stats[k].TokenID := i;
    Stats[k].Count := Counts[i];
    Inc(k);
  end;
end;

// Sort by descending count.
procedure SortMergedTokenStatsByCount(var Stats: TMergedTokenStats);
var
  i, j: Integer;
  Temp: TMergedTokenStat;
begin
  for i := 0 to High(Stats) - 1 do
    for j := i + 1 to High(Stats) do
      if Stats[j].Count > Stats[i].Count then begin
        Temp := Stats[i];
        Stats[i] := Stats[j];
        Stats[j] := Temp;
      end;
end;

{ Report Statistics }
// Report N most frequent merged tokens.
procedure ReportTopMergedTokens(const Stats: TMergedTokenStats;
  const SymbolTable: TSymbolTable; N: Integer);
var
  i, Limit: Integer;
  S: String;
begin
  Writeln('--- Top ', N, ' Most Frequent Merged Tokens ---');

  if Length(Stats) = 0 then Exit;

  Limit := N;
  if Limit > Length(Stats) then
    Limit := Length(Stats);

  for i := 0 to Limit - 1 do begin
    S := SymbolTable[Stats[i].TokenID];
    Writeln(i + 1:4, '  ID=', Stats[i].TokenID:6, '  Count=',
      Stats[i].Count:8, '  Symbol="', S, '"');
  end;
  Writeln;
end;

// Report singleton merged tokens.
procedure ReportSingletonMergedTokens(const Stats: TMergedTokenStats);
var
  i, Singletons: Integer;
begin
  Singletons := 0;

  for i := 0 to High(Stats) do
    if Stats[i].Count = 1 then
      Inc(Singletons);

  Writeln('Merged tokens used only once: ', Singletons);
end;

// Report merged tokens never used.
procedure ReportUnusedMergedTokens(const Stats: TMergedTokenStats);
var
  i, Unused: Integer;
begin
  Unused := 0;

  for i := 0 to High(Stats) do
    if Stats[i].Count = 0 then
      Inc(Unused);

  Writeln('Merged tokens never used: ', Unused);
end;

// Report coverage of top merges.
procedure ReportTopMergeCoverage(const Stats: TMergedTokenStats; TopN: Integer);
var
  i, Limit: Integer;
  TotalMergedInstances, TopMergedInstances: Integer;
  Coverage: Single;
begin
  TotalMergedInstances := 0;
  for i := 0 to High(Stats) do
    Inc(TotalMergedInstances, Stats[i].Count);

  Limit := TopN;
  if Limit > Length(Stats) then
    Limit := Length(Stats);

  TopMergedInstances := 0;
  for i := 0 to Limit - 1 do
    Inc(TopMergedInstances, Stats[i].Count);

  if TotalMergedInstances > 0 then
    Coverage := 100.0 * TopMergedInstances / TotalMergedInstances
  else
    Coverage := 0.0;

  Writeln('Top ', Limit, ' merged tokens account for ', TopMergedInstances, ' / ', TotalMergedInstances,
    ' merged-token instances = ', Coverage:0:2, '%');
end;

// Report token statistics.
procedure ReportTokenUsageStatistics;
var
  Counts: TIVector;
  Stats: TMergedTokenStats;
  FirstMergedToken: Integer;
begin
  Writeln('--- Token Statistics ---');
  Writeln('Merged token instances: ', MergedInstances);
  Writeln('Unmerged token instances: ', UnmergedInstances);
  Writeln('Mean token length: ', nCorpus / nTokenizedCorpus: 6: 4);
  CountTokenUsage(TokenizedCorpus, Length(SymbolTable), Counts);
  BuildMergedTokenStats(Counts, FirstMergedToken, Stats);
  SortMergedTokenStatsByCount(Stats);

  ReportTopMergedTokens(Stats, SymbolTable, 30);
  ReportUnusedMergedTokens(Stats);
  ReportSingletonMergedTokens(Stats);
  ReportTopMergeCoverage(Stats, 30);
end;

// Report basic statistics (time, file names).
procedure ReportBasicStatistics;
var
  i: Integer;
begin
  writeln;
  Writeln('--- File Information ---');
  writeln('Files used in symbol table: ');
  for i := 0 to High(CorpusFileNames) do
    writeln(CorpusFileNames[i], '  ');
  writeln;

  Writeln('--- Time Statistics ---');
  writeln('Start time: ', DateTimetoStr(t0), '     End time: ', DateTimeToStr(t1));
  Writeln('Total elapsed time: ', Hours, ' hours, ', Mins, ' min ', Secs: 4: 4, ' sec');
end;

// Report BPE statistics.
procedure ReportBPEStatistics;
begin
  Writeln('--- BPE Statistics ---');
  Writeln('Original text size (bytes/tokens): ', nCorpus);
  Writeln('Encoded text size (bytes/tokens): ', nTokenizedCorpus);
  Writeln('Compression ratio: ', nCorpus   / nTokenizedCorpus:0: 4);
  if not FromSymbolTable then
    Writeln('Tokens per second: ', nCorpus / (ElapsedMS / 1000): 6: 4);
  writeln;
  end;

// Report all statistics.
procedure ReportStatistics;
begin
  CalculateTimeStatistics;
  ReportBasicStatistics;
  if VerboseTokenize and (TextRec(Output).Handle = StdOutputHandle) then
    Pause;
  ReportBPEStatistics;
  ReportTokenUsageStatistics;
  Pause;
end;

{ Save data from tokenization }
// Save metadata.
procedure SaveMetaData(const MetaFileName: String);
var
  SaveOut: Text;
begin
  // Save current Output.
  SaveOut := Output;

  // Redirect Output to F.
  Assign(Output, MetaFileName);
  Rewrite(Output);

  ReportStatistics;

  // Restore Output to console.
  Close(Output);
  Output := SaveOut;

  writeln('File ', MetaFileName, ' successfully saved.');
  writeln;
end;

// Save the output token list to a .bin file.
procedure WriteTokenList(const Part: TPart = B);
var
  i, iB, iE: Integer;
begin
    Case Part of
    B: begin
      iB := 0;
      iE := 99;
    end;
    E: begin
      iB := High(TokenizedCorpus) - 99;
      iE := High(TokenizedCorpus);
    end;
    F: begin
      iB := 0;
      iE := High(TokenizedCorpus);
    end;
  end;

  write('Tokenized Corpus, ');
  Case Part of
    B: write('First 100 bytes: ');
    E: write('Last 100 bytes: ');
    F: write('All bytes: ');
  end;
  writeln;

  for i := ib to iE do
    Write(TokenizedCorpus[i], ' ');
  writeln;
  writeln('Tokenized corpus length =  ', Length(TokenizedCorpus));
  Pause
end;

// Save the output tokenized corpus to a .bin file.
procedure SaveTokenList(const BinFileName: String);
var
  F: file of Int32;
  v: Int32;
  i: Integer;
begin
  AssignFile(F, BinFileName);
  Rewrite(F);

  for i := 0 to High(TokenizedCorpus) do begin
    v := TokenizedCorpus[i];
    Write(F, v);
  end;

  CloseFile(F);
  writeln('File ', BinFileName, ' successfully saved.');
end;

{ Load tables }
// Load tokenized corpus from a .bin file.
procedure LoadTokenList(const BinFileName: String);
var
  F: file of Int32;
  v: Int32;
  i, Count: Integer;
begin
  AssignFile(F, BinFileName);
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
  Writeln('Loaded ', Count, ' tokens from ', BinFileName);
end;

// Reconstruct corpus from tokenized corpus.
procedure DetokenizeToDisplay(const TokenizedCorpus: TIVector; const Part: TPart = B);
var
  i, iB, iE, t, symIndex: Integer;
begin
  Case Part of
    B: begin
      iB := 0;
      iE := 499;
    end;
    E: begin
      iB := High(TokenizedCorpus) - 499;
      iE := High(TokenizedCorpus);
    end;
    F: begin
      iB := 0;
      iE := High(TokenizedCorpus);
    end;
  end;

  write('Detokenized Corpus, ');
  Case Part of
    B: write('First 500 bytes: ');
    E: write('Last 500 bytes: ');
    F: write('All bytes: ');
  end;
  writeln;
  Pause;
  for i := iB to iE do begin
    t := TokenizedCorpus[i];

    if t < 256 then begin
      // Raw byte.
      Write(Char(t));
    end
    else if t < 260 then begin
      // Special tokens.
      case t of
        256: Write('<BOS>');
        257: Write('<EOS>');
        258: Write('<UNK>');
        259: Write('<PAD>');
      end;
    end
    else begin
      // Symbol table entry.
      symIndex := t - 260;
      Write(SymbolTable[symIndex]);   // Or just SymbolTable[t]?? Says ChaptGPT.
    end;
  end;

  writeln;
end;

function CompareByLength(List: TStringList; Index1, Index2: Integer): Integer;
begin
  // Sort longest first
  Result := Length(List[Index2]) - Length(List[Index1]);
end;

// Run the tokenizer.
procedure RunWesTokenize(var Corpus: TBVector; const SymbolTable: TSymbolTable);
begin
  // Timing.
  t0 := Now;       // Start of timing for entire tokenization;
  StopTime := 0;   // Time to subtract from timing.

  // Create the TokenList.
  writeln('Maximum symbols = ', MaxVocab, '. Maximum merges = ', MaxMerges, '. Maximum pair counts = ', MaxPairCount, '. Tokenizing...');
  writeln('X = Exit program. B = Break out of merge loop. V = toggle Verbose mode. P = Program information. M = Merging information. Merging...');
  //BuildTokenListFromCorpus(Corpus);

  // Create the tokenized corpus.
  TokenizeFromSymbolTable(FileName, Corpus);
  //procedure TokenizeFromSymbolTable(const TextFileName: string; var Corpus: TBVector);

  // Timing.
  t1 := Now;

  if ShowTokenWork and VerboseTokenize then begin
    Writeln('---  Token Frequencies ---');
    CountSymbols(SymbolTable);
  end;

  nSymbols := Length(SymbolTable);
  nVocab := nSymbols;

  // Report statistics.
  if VerboseTokenize then begin
    ReportStatistics;
    Pause;
  end;

  // Create new directory and stamps for saving files.
  Stamp := FormatDateTime('yyyy-mm-dd_hhnnss', Now);
  CreateDir(WorkingName + Stamp);
  ChDir(WorkingName + Stamp);

  // Verify by reconstructing.
  if ShowVerification and VerboseTokenize and DisplayCorpus then begin
    writeln('--- Reconstructed Corpus ---');
  DetokenizeToDisplay(TokenizedCorpus, B);
end;

  writeln('End of tokenization.');
  Pause;
  //nTokens := nTokenizedCorpus;    // For embedding, need nTokens.

  if VerboseTokenize then Begin
    writeln('First 150 tokens of tokenized corpus:');
    for i := 0 to 149 do
      write(TokenizedCorpus[i], ' ');
    writeln;
    Pause;
  end;
end;

end.

