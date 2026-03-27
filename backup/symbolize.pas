unit Symbolize;

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
  IOHandler,
  SysUtils;

type
  PTokenNode = ^TTokenNode;            // Doubly-linked list.
  TTokenNode = record                  // Each node as a token, an integer corresponding to a symbol.
    Tok: Integer;
    Prev, Next: PTokenNode;
  end;

  type
  TPairSlotState = (psEmpty, psUsed);

  TPairHashEntry = record
    A, B: Integer;
    Count: Integer;
    State: TPairSlotState;
  end;

  TPairHash = record
    Entries: array of TPairHashEntry;
    Capacity: Integer;
    Used: Integer;
  end;

  // Old pair count code.
  TPairCount = record                  // Record of pair counts.
    A, B: Integer;                     // A and B are the pair.
    Count: Integer;                    // Count is how often they occur.
  end;
  TPairCounts = array of TPairCount;   // Array of pair counts.

  TMerge = record                      // Record for merger of two nodes.
    A, B: Integer;                     // Original pair.
    NewSym: Integer;                   // New integer for symbol.
  end;
  TMergeArray = array of TMerge;       // Array of merges.

var
  StartSymbol: Integer = 260;                    // UTF-8 0.255, BOS, EOS, PAD, UNK is 259.
  nCorpus: Integer;
  ElapsedMS, MElapsedMS: Int64;                  // For timing.
  MHours, Hours, MMIns, Mins: Int64;             // For timing.
  Secs, MSecs: Double;                           // For timing.
  BOS, EOS, PAD, UNK: Integer;                   // Extra symbols for control.
  Head, Tail: PTokenNode;                        // Start and end node of list of tokens.
  MergeCount: Integer;                           // Maximum allowed number of merges and actual number.
  Merges: TMergeArray;                           // Array recording the merges.
  Magic: array[0..3] of Char = ('S', 'Y', 'M', 'T');  // For saving symbol table.
  MergedTypes, UnmergedTypes: Integer;

procedure ReadFileBytes(const FileName: String; var OneCorpus: TBVector);
procedure ReportStatistics;
procedure RunSymbolize(const Corpus: TBVector);

implementation

// Apply a learned symbol table to a raw byte corpus.
// Input:
//   SymbolTable: array of learned symbols, each symbol is an array of bytes.
//   nSymbols (aka nVocab): number of entries in SymbolTable.
//   Corpus: raw byte text.
// Output:
//   TokenizedCorpus: dynamic array of token IDs.

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
// To prevent special characters from merging.
function IsSpecial(T: Integer): Boolean;
begin
  Result := (T = BOS) or (T = EOS) or (T = PAD) or (T = UNK);
end;

// Build the initial token linked list from the Corpus.
procedure BuildTokenListFromCorpus(const Corpus: TBVector);
var
  i: Integer;
  Node, Prev: PTokenNode;
begin
  Head := nil;
  Tail := nil;
  Prev := nil;

  for i := 0 to High(Corpus) do begin
    New(Node);
    Node^.Tok := Corpus[i];
    Node^.Prev := Prev;
    Node^.Next := nil;

    if Prev <> nil then
      Prev^.Next := Node
    else
      Head := Node;

    Prev := Node;
  end;

  Tail := Prev;
end;

// Init hash code.
procedure InitPairHash(var H: TPairHash; InitialCapacity: Integer);
var
  i: Integer;
begin
  if InitialCapacity < 16 then
    InitialCapacity := 16;

  H.Capacity := InitialCapacity;
  H.Used := 0;
  SetLength(H.Entries, H.Capacity);

  for i := 0 to H.Capacity - 1 do begin
    H.Entries[i].A := 0;
    H.Entries[i].B := 0;
    H.Entries[i].Count := 0;
    H.Entries[i].State := psEmpty;
  end;
end;

function HashPair(A, B, Capacity: Integer): Integer;
var
  H: QWord;
begin
  H := QWord(Cardinal(A)) * 1000003 + QWord(Cardinal(B));
  Result := Integer(H mod QWord(Capacity));
end;

function FindSlot(const H: TPairHash; A, B: Integer): Integer;
var
  Idx: Integer;
begin
  Idx := HashPair(A, B, H.Capacity);

  while H.Entries[Idx].State = psUsed do begin
    if (H.Entries[Idx].A = A) and (H.Entries[Idx].B = B) then Exit(Idx);

    Idx := (Idx + 1) mod H.Capacity;
  end;

  Result := Idx;
end;

procedure PairIncHash(var H: TPairHash; A, B: Integer);
var
  Idx: Integer;
begin
  Idx := FindSlot(H, A, B);

  if H.Entries[Idx].State = psUsed then
    Inc(H.Entries[Idx].Count)
  else begin
    H.Entries[Idx].State := psUsed;
    H.Entries[Idx].A := A;
    H.Entries[Idx].B := B;
    H.Entries[Idx].Count := 1;
    Inc(H.Used);
  end;
end;

procedure PairDecHash(var H: TPairHash; A, B: Integer);
var
  Idx: Integer;
begin
  Idx := FindSlot(H, A, B);

  if (H.Entries[Idx].State = psUsed) and
     (H.Entries[Idx].A = A) and
     (H.Entries[Idx].B = B) then
  begin
    if H.Entries[Idx].Count > 0 then
      Dec(H.Entries[Idx].Count);
  end;
end;

function FindBestPairHash(const H: TPairHash; out A, B: Integer): Integer;
var
  i, Max: Integer;
begin
  Max := 0;
  A := -1;
  B := -1;

  for i := 0 to H.Capacity - 1 do
    if (H.Entries[i].State = psUsed) and (H.Entries[i].Count > Max) then begin
      Max := H.Entries[i].Count;
      A := H.Entries[i].A;
      B := H.Entries[i].B;
    end;

  Result := Max;
end;

// Init pairs hash routine.
procedure InitPairHashFromList(Head: PTokenNode; var H: TPairHash);
var
  Cur: PTokenNode;
begin
  Cur := Head;

  while (Cur <> nil) and (Cur^.Next <> nil) do begin
    if not (IsSpecial(Cur^.Tok) or IsSpecial(Cur^.Next^.Tok)) then
      PairIncHash(H, Cur^.Tok, Cur^.Next^.Tok);

    Cur := Cur^.Next;
  end;
end;

// Update pairs hash.
procedure UpdatePairsForMergeHash(Node: PTokenNode; NewTok: Integer; var H: TPairHash);
var
  A, B: Integer;
begin
  if (Node = nil) or (Node^.Next = nil) then Exit;

  A := Node^.Tok;
  B := Node^.Next^.Tok;

  // Remove (A, B).
  PairDecHash(H, A, B);

  // Remove (Prev, A).
  if Node^.Prev <> nil then
    PairDecHash(H, Node^.Prev^.Tok, A);

  // Remove (B, Next).
  if Node^.Next^.Next <> nil then
    PairDecHash(H, B, Node^.Next^.Next^.Tok);

  // Add (Prev, NewTok).
  if Node^.Prev <> nil then
    PairIncHash(H, Node^.Prev^.Tok, NewTok);

  // Add (NewTok, Next).
  if Node^.Next^.Next <> nil then
    PairIncHash(H, NewTok, Node^.Next^.Next^.Tok);
end;

{ Merge process in linked list }
// Merge two nodes in token linked list.
procedure MergeAt(var Head, Tail: PTokenNode; Node: PTokenNode; NewTok: Integer);
var
  Right: PTokenNode;
begin
  Right := Node^.Next;
  if Right = nil then Exit;

  // If merging away the tail, update Tail.
  if Right = Tail then
    Tail := Node;

  // Replace Node + Right with NewTok.
  Node^.Tok := NewTok;
  Node^.Next := Right^.Next;

  if Right^.Next <> nil then
    Right^.Next^.Prev := Node;

  Dispose(Right);
end;

// Merge for pairs hash.
procedure MergeAllPairsHash(var Head, Tail: PTokenNode; A, B, NewTok: Integer; var H: TPairHash);
var
  Cur: PTokenNode;
begin
  Cur := Head;

  while (Cur <> nil) and (Cur^.Next <> nil) do begin
    if not (IsSpecial(Cur^.Tok) or IsSpecial(Cur^.Next^.Tok)) then begin
      if (Cur^.Tok = A) and (Cur^.Next^.Tok = B) then begin
        UpdatePairsForMergeHash(Cur, NewTok, H);
        MergeAt(Head, Tail, Cur, NewTok);
        Cur := Cur^.Next;
      end
      else
        Cur := Cur^.Next;
    end
    else
      Cur := Cur^.Next;
  end;
end;

// Record the merge in the Merges array.
procedure RecordMerge(var Merges: TMergeArray; MergeIndex, A, B, NewSym: Integer);
begin
  if MergeIndex >= Length(Merges) then
    SetLength(Merges, MergeIndex + 1);

  Merges[MergeIndex].A := A;
  Merges[MergeIndex].B := B;
  Merges[MergeIndex].NewSym := NewSym;
end;

{ Symbol Table }
// Initialize the symbol table with special characters.
procedure InitSymbolTable;
var
  i: Integer;
begin
  // 0..255 = bytes.
  SetLength(SymbolTable, 256);
  for i := 0 to 255 do
    SymbolTable[i] := Chr(i);

  // Add BOS. 256.
  BOS := Length(SymbolTable);
  SetLength(SymbolTable, BOS + 1);
  SymbolTable[BOS] := '<BOS>';

  // Add EOS. 257.
  EOS := Length(SymbolTable);
  SetLength(SymbolTable, EOS + 1);
  SymbolTable[EOS] := '<EOS>';

  // Add PAD. 258.
  PAD := Length(SymbolTable);
  SetLength(SymbolTable, PAD + 1);
  SymbolTable[PAD] := '<PAD>';

  // Add UNK. 259.
  UNK := Length(SymbolTable);
  SetLength(SymbolTable, UNK + 1);
  SymbolTable[UNK] := '<UNK>';
end;

// After performing a merge, add a new merge symbol to the symbol table.
procedure AddMergeSymbol(NewTok, A, B: Integer);
begin

  if (A < 0) or (A >= Length(SymbolTable)) then
    writeln('Invalid symbol A=', A);

  if (B < 0) or (B >= Length(SymbolTable)) then
    writeln('Invalid symbol B=', B);

  // Debugging.
  // Writeln('AddMergeSymbol: NewTok=', NewTok, ' A=', A, ' B=', B, ' Len=', Length(Table));

  // Ensure the table is large enough.
  if NewTok >= Length(SymbolTable) then
    SetLength(SymbolTable, NewTok + 1);

  // Represent the new token as concatenation of its components.
  if Length(SymbolTable[A]) + Length(SymbolTable[B]) < 4096 then
    SymbolTable[NewTok] := SymbolTable[A] + SymbolTable[B]
  else
    SymbolTable[NewTok] := '';  // lazy expansion

end;

{ Apply the BPE encoder }
// Main training loop, traverse the merges.
procedure TrainBPEHash(var Head, Tail: PTokenNode; MaxMerges: Integer;
  var MergeCount, StartSymbol: Integer);
var
  m, BestCount, A, B: Integer;
  H: TPairHash;

  procedure ReadMergeIfKeyPressed;
  var
    key: Char;
  begin
    key := CheckForControlKey;
    case key of
      'x', 'X':
        begin
          Writeln('Exit requested. Stopping execution.');
          Pause;
          Halt;
        end;
      'b', 'B':
        begin
          Writeln('Break requested. Exiting loop.');
          Pause;
          BestCount := 0;   // Causes outer loop to stop.
        end;
      'v', 'V':
        begin
          VeryVerbose := not VeryVerbose;
          Writeln('Very verbose mode: ', VeryVerbose);
        end;
      'i', 'I':
        begin
          Writeln;
          ReportInfo;
          Pause;
        end;
      'p', 'P':
        begin
          Pause;
        end;
      'm', 'M':
        begin
          Writeln;
          Writeln('Maximum symbols = ', MaxVocab, '. Maximum merges = ', MaxMerges,
            '. Hash capacity = ', H.Capacity, '. Used slots = ', H.Used, '. Best count = ', BestCount, '.');
          Write(DateTimeToStr(Now), '  X = Exit program. B = Break out of merge loop. V = toggle Verbose mode.');
          Writeln('  I = program Information. P = Pause. M = Merging information. S = maximum Symbols. Merging...');
        end;
      's', 'S':
        begin
          Writeln;
          Write('Current maximum symbols = ', MaxVocab, '. Enter new maximum symbols: ');
          ReadLn(MaxVocab);
        end;
    end;
  end;

begin
  MergeCount := 0;

  Write(DateTimeToStr(Now), '  X = Exit program. B = Break out of merge loop. V = toggle Verbose mode.');
  Writeln('  P = Program information. M = Merging information. S = maximum Symbols. Merging...');
  Writeln;

  if ShowMergeWork then
    Writeln('--- List of Merges (Hash) ---');

  for m := 1 to MaxMerges do begin
    if PauseIfKeyPressed then
      ReadMergeIfKeyPressed;

    // Rebuild pair counts from current token list.
    InitPairHash(H, MaxPairCount * 2 + 1024);
    InitPairHashFromList(Head, H);

    // Optional: save partial symbol table.
    if SavePartialSymbolTable then
      if (Length(SymbolTable) mod PartialSymbolTableTrigger) = 0 then begin
        ChDir(WorkingDir);
        SaveSymbolTable(WorkingDir + FormatDateTime('yyyy-mm-dd_hhnnss' + '.sym', Now), SymbolTable);
        ChDir('..');
      end;

    // Stop if hash table got too full.
    if H.Used > MaxPairCount then begin
      writeln;
      Writeln('Stopping: pair table exceeded ', MaxPairCount, ' entries.');
      Break;
    end;

    BestCount := FindBestPairHash(H, A, B);

    // Stop if no useful merges remain.
    if BestCount < 2 then begin
      writeln;
      Writeln('Stopping: no more valid merges at iteration ', m, '.');
      Break;
    end;

    // Stop if symbol table is full.
    if Length(SymbolTable) >= MaxVocab then begin
      writeln;
      Writeln('Stopping: symbol table reached ', MaxVocab, ' entries.');
      Break;
    end;

    // Perform merge.
    MergeAllPairsHash(Head, Tail, A, B, StartSymbol, H);

    AddMergeSymbol(StartSymbol, A, B);
    RecordMerge(Merges, MergeCount, A, B, StartSymbol);

    Inc(MergeCount);
    Inc(StartSymbol);

    if ShowMergeWork then begin
      Write(MergeCount, ' Merged (', A:5, ',', B:5, ') -> (', StartSymbol - 1:5, ') #', BestCount);
      if (MergeCount mod 4) = 0 then
        Writeln
      else
        Write('  |  ');
    end;
  end;

  Writeln('Hash tokenization complete. Total merges: ', MergeCount, '.');
  Pause;
end;

{ Display routines }
// Display all symbols in the Corpus with their frequency.
{procedure DisplayAllTokenFrequencies(const Corpus: TBVector);
var
  Counts: array of Integer;
  TST: String;
  i, j, k, S, LS, MaxSymbol: Integer;
  TokenList: TTokenCounts;
  Temp: TTokenCount;
begin

  // Find the maximum symbol value.
  MaxSymbol := 0;
  for i := 0 to High(Corpus) do
    if Corpus[i] > MaxSymbol then MaxSymbol := Corpus[i];

  // Initialize counts array.
  SetLength(Counts, MaxSymbol + 1);
  for i := 0 to MaxSymbol do
    Counts[i] := 0;

  // Count occurrences of each symbol.
  for i := 0 to High(Corpus) do
    Counts[Corpus[i]] := Counts[Corpus[i]] + 1;

  // Build list of tokens with count > 0.
  SetLength(TokenList, 0);
  for i := 0 to MaxSymbol do
    if Counts[i] > 0 then begin
      SetLength(TokenList, Length(TokenList) + 1);
      TokenList[High(TokenList)].Symbol := i;
      TokenList[High(TokenList)].Count := Counts[i];
    end;

  // Sort descending by count.
  for i := 0 to High(TokenList) - 1 do
    for j := i + 1 to High(TokenList) do
      if TokenList[i].Count < TokenList[j].Count then begin
        Temp := TokenList[i];
        TokenList[i] := TokenList[j];
        TokenList[j] := Temp;
      end;

  // Print all symbols with frequency.
  for i := 0 to High(TokenList) do begin
      writeln(i: 4, '  ', SymbolTable[TokenList[i].Symbol], '   ', TokenList[i].Count);
  {  S := TokenList[i].Symbol;
    LS := Length(SymbolTable[S]);
    TST := SymbolTable[S];        // TST is temporary SymbolTable character.
    for j := 1 to LS do begin     // Used for displaying below.
      k := Ord(TST[j]);
      // Unknown character is a hex, not a dot, also char 183, for display.
      if (k >= 32) and (k <= 126) then
        Write(Chr(k))
      else
        Write('\x', IntToHex(k, 2));
      // if (k < 32) or (k > 126) then TST[j] := Chr(183);
    end;
    Write(i: 5, S: 5, ' ': (10 - LS), '*', TST, '*', TokenList[i].Count: 5, '          ');
    if (i mod 4 = 3) then writeln;}
  end;
end;}

{ Computations and reports }
// Calculate time statistics.
procedure CalculateTimeStatistics;
begin
  // Total elapsed time.
  ElapsedMS := MilliSecondsBetween(t0, t1) - Round(StopTime);
  Hours := ElapsedMS div 3600000;
  Mins := ElapsedMS div 60000;
  Secs := (ElapsedMS mod 60000) / 1000.0;
  // Merge eotal elapsed time.
  MElapsedMS := MilliSecondsBetween(Mt0, Mt1) - Round(StopTime);
  MHours := MElapsedMS div 3600000;
  MMins := MElapsedMS div 60000;
  MSecs := (MElapsedMS mod 60000) / 1000.0;
end;

// Calculate and symbols statistics.
procedure SymbolStats;
var
  n, i, j, L, MinLen, MaxLen, SumLen: Integer;
  Lengths, Histogram: TIVector;
  MaxPossibleLen: Integer;
  Median: Single;
begin
  n := Length(SymbolTable);
  if n = 0 then begin
    WriteLn('Symbol table is empty.');
    Exit;
  end;

  writeln('--- Symbols Statistics ---');
  writeln('Number of raw byte symbols: ', 256);
  writeln('Number of special symbols: ', 4);
  writeln('Number of merged symbols: ', nSymbols - 260);

  { --- First pass: compute lengths, min, max, sum --- }
  SetLength(Lengths, n);

  MinLen := MaxInt;
  MaxLen := 0;
  SumLen := 0;

  for i := 0 to n - 1 do begin
    L := Length(SymbolTable[i]);  // Byte length.
    Lengths[i] := L;

    if L < MinLen then MinLen := L;
    if L > MaxLen then MaxLen := L;

    SumLen := SumLen + L;
  end;

  // Min / Max.
  WriteLn('Minimum symbol length: ', MinLen);
  WriteLn('Maximum symbol length: ', MaxLen);

  // Histogram.
  MaxPossibleLen := MaxLen;
  SetLength(Histogram, MaxPossibleLen + 1);
  for i := 0 to MaxPossibleLen do
    Histogram[i] := 0;

  for i := 0 to n - 1 do
    Inc(Histogram[Lengths[i]]);

  WriteLn;
  WriteLn('Histogram of symbol lengths:');
  for i := 0 to MaxPossibleLen do
    if Histogram[i] > 0 then
      WriteLn('Length ', i: 2, ': ', Histogram[i]);

  // --- Median ---
  // Sort the Lengths array
  for i := 1 to n - 1 do begin
    L := Lengths[i];
    j := i - 1;
    while (j >= 0) and (Lengths[j] > L) do begin
      Lengths[j + 1] := Lengths[j];
      Dec(j);
    end;
    Lengths[j + 1] := L;
  end;

  if (n mod 2) = 1 then
    Median := Lengths[n div 2]
  else
    Median := 0.5 * (Lengths[n div 2 - 1] + Lengths[n div 2]);

  WriteLn;
  WriteLn('Mean symbol length: ', SumLen / n: 0: 4);
  WriteLn('Median symbol length: ', Median: 0: 4);
  writeln('Mean tokens per symbol (compression): ', (nCorpus / nSymbols): 0: 4);
end;

// Calculate and report longest symbols.
procedure ReportSymbolLengths;
var
  i, MaxLen, MaxIndex, SumLen: Integer;
  SymbolLengths: array[1..10] of Integer;
begin
  MaxLen := 0;
  MaxIndex := -1;
  SumLen := 0;
  FillChar(SymbolLengths, SizeOf(SymbolLengths), 0);

  for i := 0 to High(SymbolTable) do begin
    if Length(SymbolTable[i]) > MaxLen then begin
      MaxLen := Length(SymbolTable[i]);
      MaxIndex := i;
    end;
    SumLen := SumLen + Length(SymbolTable[i]);
    if (Length(SymbolTable[i]) <= 9) then
      Inc(SymbolLengths[Length(SymbolTable[i])])
    else
      Inc(SymbolLengths[10]);
  end;

  if maxIndex >= 0 then  begin
    writeln('Longest symbol:');
    writeln('  Index: ', maxIndex);
    writeln('  Length: ', maxLen);
    writeln('  Value: "', SymbolTable[maxIndex], '"');
  end;
end;

{ Report Statistics }
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
  Writeln('Number of symbols: ', nSymbols);
  if not FromSymbolTable then begin
    Writeln('Elapsed time applying merges: ', MHours, ' hours, ', Mmins, ' min ', Msecs: 4: 4, ' sec');
  end;
  Writeln('Original text size (bytes/tokens): ', nCorpus);
  if not FromSymbolTable then begin
    Writeln('Tokens per second (total): ', nCorpus / (ElapsedMS / 1000): 6: 4);
    Writeln('Tokens per second (merging): ', nCorpus / (MElapsedMS / 1000): 6: 4);
    writeln;
  end;
end;

// Report all statistics.
procedure ReportStatistics;
begin
  CalculateTimeStatistics;
  ReportBasicStatistics;
  SymbolStats;
  ReportSymbolLengths;
  if VerboseTokenize and (TextRec(Output).Handle = StdOutputHandle) then
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

// Save merge table.
procedure SaveMergeTable(const Merges: TMergeArray; MergeFileName: String);
var
  F: file;
  i, n: Integer;
begin
  Assign(F, MergeFileName);
  Rewrite(F, 1);

  n := Length(Merges);
  BlockWrite(F, n, SizeOf(n));

  for i := 0 to n - 1 do begin
    BlockWrite(F, Merges[i].A, SizeOf(Integer));
    BlockWrite(F, Merges[i].B, SizeOf(Integer));
    BlockWrite(F, Merges[i].NewSym, SizeOf(Integer));
  end;

  Close(F);
  writeln('File ', MergeFileName, ' successfully saved.');
end;

// Run the tokenizer.
procedure RunSymbolize(const Corpus: TBVector);
begin
  // Timing.
  t0 := Now;       // Start of timing for entire tokenization;
  StopTime := 0;   // Time to subtract from timing.

  // Create the TokenList.
  writeln('Maximum symbols = ', MaxVocab, '. Maximum merges = ', MaxMerges, '. Maximum pair counts = ', MaxPairCount, '.');
  writeln('X = Exit program. B = Break out of merge loop. V = toggle Verbose mode. P = Program information. M = Merging information. Merging...');
  BuildTokenListFromCorpus(Corpus);

  nCorpus := Length(Corpus);

  // First merge symbol is StartSymbol, 260.
  InitSymbolTable;

  // Run BPE.
  Mt0 := Now;
  TrainBPEHash(Head, Tail, MaxMerges, MergeCount, StartSymbol);
  Mt1 := Now;

  // Timing.
  t1 := Now;

  nSymbols := Length(SymbolTable);
  // Display symbol table.
  if VerboseTokenize then
    DisplayByteSymbolTable(SymbolTable);

  // Report statistics.
  if VerboseTokenize then
    ReportStatistics;

  // Save various files.
  if SaveFiles then begin
    ChDir(WorkingDir);
    writeln('--- Saving Files ---');
    SaveSymbolTable(WorkingName + '.sym', SymbolTable);
    SaveMergeTable(Merges, WorkingName + '.mer');
    SaveMetaData(WorkingName + '.meta');
    ChDir('..');
  end;

  writeln('End of symbolization procedure.');
  Pause;
end;

end.

