unit WesTokenize;

{$mode ObjFPC}{$H+}{$I proprietary.txt}

{ WesChat, Version 1.2, begun January 10, 2026, by Wesley R. Parsons, wespar@bellouth.net, www.wespar.com.}
{ Note: Edited 3/21/2026 5:07 pm }

interface

uses
  Classes,
  Crt,
  DateUtils,
  Display,
  FileUtil,
  Global,
  IOHandler,
  SysUtils;

type
  TTokenCount = record                 // Records count of tokens.
    Symbol: Integer;                   // Symbol for token.
    Count: Integer;                    // Number of times it occurs.
  end;
  TTokenCounts = array of TTokenCount; // Array of token counts.

// Use in Trie.
  PTrieNode = ^TTrieNode;
  TTrieNode = record
    Children: array[0..255] of PTrieNode; // ASCII.
    TokenID: Integer;                     // -1 if not terminal.
  end;

// Use in token statistics.
  TMergedTokenStat = record
    TokenID: Integer;
    Count: Integer;
  end;

  TMergedTokenStats = array of TMergedTokenStat;
var
  StartSymbol: Integer = 260;                    // UTF-8 0.255, BOS, EOS, PAD, UNK is 259.
  TokenizedCorpus: TIVector;
  BOS, EOS, PAD, UNK: Integer;                   // Extra symbols for control.
  ElapsedMS: Int64;                              // For timing.
  Hours, Mins: Int64;                            // For timing.
  Secs, MSecs: Double;                           // For timing.
  FileName, Reconstructed: String;               // Saving data.
  Magic: array[0..3] of Char = ('S', 'Y', 'M', 'T');  // For saving symbol table.
  TrieHead: PTrieNode = nil;                     // Nodes for Trie.
  MergedTypes, UnmergedTypes: Integer;
  MergedInstances, UnmergedInstances: Integer;
  i: Integer;

procedure WriteTokenList(const Part: TPart = B);
procedure BuildTrie(out Root: PTrieNode);
function MatchLongest(root: PTrieNode; const text: TBVector; startPos: Integer;
  out tokenID, matchLen: Integer): Boolean;
procedure DetokenizeToDisplay(const TokenizedCorpus: TIVector; const Part: TPart = B);
procedure ReportStatistics;
procedure RunWesTokenize(const Corpus: TBVector; var TokenizedCorpus: TIVector);

implementation

// Trie procedures: Insert trie symbol.
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

// Trie procedure: Build trie.
procedure BuildTrie(out Root: PTrieNode);
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

// Trie procedure: Match longest.
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
procedure TokenizeFromSymbolTable(const TextFileName: string; const Corpus: TBVector);
var
  i, BestSym, BestLen: Integer;
begin
//  if FileExists(TextFileName) then      ```don't read again!!
  //  ReadFileBytes(TextFileName, Corpus);

  nCorpus := Length(Corpus);
  SetLength(TokenizedCorpus, 1);
  TokenizedCorpus[0] := 256;
  i := 0;

  BuildTrie(TrieHead);

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

  SetLength(TokenizedCorpus, Length(TokenizedCorpus) + 1);
  TokenizedCorpus[Length(TokenizedCorpus) - 1] := 257;

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

// Calculate number of symbol types and token instances.
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
procedure CountSymbols;
var
  Counts, Index: TIVector;
  i, j, k, N, TmpIndex: Integer;
begin
  // Allocate and zero Counts.
  SetLength(Counts, Length(SymbolTable));
  FillChar(Counts[0], Length(Counts) * SizeOf(Counts[0]), 0);

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
procedure ReportTopMergedTokens(const Stats: TMergedTokenStats; N: Integer);
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
    S := CleanUpSymbol(SymbolTable[Stats[i].TokenID]);
    Writeln(i + 1:4, '  ID=', Stats[i].TokenID:6, '  Count=',
      Stats[i].Count:8, '  Symbol="', S, '"');
  end;
  Writeln;
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

// Report token usage statistics.
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
  FirstMergedToken := StartSymbol;  // 260
  BuildMergedTokenStats(Counts, FirstMergedToken, Stats);
  SortMergedTokenStatsByCount(Stats);

  ReportTopMergedTokens(Stats, 30);
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

// Display the toeknized corpus.
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

// Detokenize tokenized corpus to text.
procedure DetokenizeToDisplay(const TokenizedCorpus: TIVector; const Part: TPart = B);
var
  i, iB, iE, Cutoff: Integer;
begin
  if Length(TokenizedCorpus) < 499 then
    Cutoff := Length(TokenizedCorpus) - 1
  else
    Cutoff := 499;
  Case Part of
    B: begin
      iB := 0;
      iE := Cutoff;
    end;
    E: begin
      iB := High(TokenizedCorpus) - Cutoff;
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
  for i := iB to iE do
    write(SymbolTable[TokenizedCorpus[i]]);
  writeln;
end;

// Run the tokenizer.
procedure RunWesTokenize(const Corpus: TBVector; var TokenizedCorpus: TIVector);
begin
  // Timing.
  t0 := Now;       // Start of timing for entire tokenization;
  StopTime := 0;   // Time to subtract from timing.

  // Display stats.
  writeln('Maximum symbols = ', MaxVocab, '. Maximum merges = ', MaxMerges, '. Maximum pair counts = ', MaxPairCount, '. Tokenizing...');
  writeln('X = Exit program. B = Break out of merge loop. V = toggle Verbose mode. P = Program information. M = Merging information. Merging...');
   // ''symboltable from opt 2 is zero
  DisplayByteSymbolTable(SymbolTable);
  Pause;

  // Create the tokenized corpus.
  TokenizeFromSymbolTable(FileName, Corpus);
  //procedure TokenizeFromSymbolTable(const TextFileName: string; var Corpus: TBVector);

  // Timing.
  t1 := Now;

  if ShowTokenWork and VerboseTokenize then begin
    Writeln('---  Token Frequencies ---');
    CountSymbols;
  end;

  nSymbols := Length(SymbolTable);
  nVocab := nSymbols;

  // Report statistics.
  if VerboseTokenize then begin
    ReportStatistics;
    Pause;
  end;

  If SaveFiles then begin
    // Create new directory and stamps for saving files.
    Stamp := FormatDateTime('yyyy-mm-dd_hhnnss', Now);
    CreateDir(WorkingName + Stamp);
    ChDir(WorkingName + Stamp);

    // Save TokenizedCorpus.
    SaveTokenList(TokenizedCorpus, WorkingName + Stamp + '.tok');
    ChDir('..');
  end;

  // Verify by reconstructing.
  if ShowVerification and VerboseTokenize and DisplayCorpus then begin
    writeln('--- Reconstructed Corpus ---');
  DetokenizeToDisplay(TokenizedCorpus, B);
  writeln;
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

