unit Symbolize;

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
  IOHandler,
  Math,
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

  // Lazy max-heap for pair selection.
  // The heap may contain stale counts. When the top entry is popped,
  // it is checked against the current count in TPairHash.
  TPairHeapEntry = record
    A, B: Integer;
    Count: Integer;
  end;

  TPairHeap = record
    Items: array of TPairHeapEntry;
    Count: Integer;
  end;

  TMerge = record                      // Record for merger of two nodes.
    A, B: Integer;                     // Original pair.
    NewSym: Integer;                   // New integer for symbol.
  end;
  TMergeArray = array of TMerge;       // Array of merges.

var
  StartSymbol: Integer = 260;                    // UTF-8 0.255, BOS, EOS, PAD, UNK is 259.
  FinalTokenCount: Integer;                      // Number of tokens by traversing nodes.
  ElapsedMS: Int64;                              // For timing.
  Hours, Mins: Int64;                            // For timing.
  Secs: Double;                                  // For timing.
  Head, Tail: PTokenNode;                        // Start and end node of list of tokens.
  MergeCount: Integer;                           // Maximum allowed number of merges and actual number.
  Merges: TMergeArray;                           // Array recording the merges.

procedure ReadFileBytes(const FileName: String; var OneCorpus: TBVector);
procedure SaveMergeTable(const Merges: TMergeArray; MergeFileName: String);
procedure SaveMetaData(const MetaFileName: String);
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
  if VerboseTokenize then
    Writeln('--- Original Corpus ---');
  for i := 0 to Size - 1 do begin
    BlockRead(F, B, 1);
    OneCorpus[i] := B;

    if VerboseTokenize then
      if DisplayEachByteRead then
        if B < 32 then
          Write('<', B, '>')
        else
          Write(Chr(B));
  end;
  CloseFile(F);

  if VerboseTokenize then begin
    Writeln('ReadByteFile: ');
    for i := 0 to Min(150, High(OneCorpus)) do
      Write(OneCorpus[i], ' ');
    Pause;
    Writeln;
  end;

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

    // Tiny Stories separator byte 254 becomes the EOS token.
    if Corpus[i] = 254 then
      Node^.Tok := EOS
    else
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

// Free the token list at end of procedure.
procedure FreeTokenList(var Head, Tail: PTokenNode);
var
  Cur, Next: PTokenNode;
begin
  Cur := Head;

  while Cur <> nil do begin
    Next := Cur^.Next;
    Dispose(Cur);
    Cur := Next;
  end;

  Head := nil;
  Tail := nil;
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

function PairAllowed(A, B: Integer): Boolean;
begin
  Result := not (IsSpecial(A) or IsSpecial(B));
end;

function PairGetCount(const H: TPairHash; A, B: Integer): Integer;
var
  Idx: Integer;
begin
  if H.Capacity <= 0 then begin
    Result := 0;
    Exit;
  end;

  Idx := FindSlot(H, A, B);

  if (H.Entries[Idx].State = psUsed) and (H.Entries[Idx].A = A) and (H.Entries[Idx].B = B) then
    Result := H.Entries[Idx].Count
  else
    Result := 0;
end;

function PairIncHash(var H: TPairHash; A, B: Integer): Integer;
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

  Result := H.Entries[Idx].Count;
end;

function PairDecHash(var H: TPairHash; A, B: Integer): Integer;
var
  Idx: Integer;
begin
  Result := 0;
  Idx := FindSlot(H, A, B);

  if (H.Entries[Idx].State = psUsed) and (H.Entries[Idx].A = A) and (H.Entries[Idx].B = B) then begin
    if H.Entries[Idx].Count > 0 then
      Dec(H.Entries[Idx].Count);
    Result := H.Entries[Idx].Count;
  end;
end;

{ Lazy pair max-heap }
procedure InitPairHeap(var Heap: TPairHeap; InitialCapacity: Integer);
begin
  if InitialCapacity < 16 then
    InitialCapacity := 16;

  SetLength(Heap.Items, InitialCapacity);
  Heap.Count := 0;
end;

function HeapEntryGreater(const L, R: TPairHeapEntry): Boolean;
begin
  if L.Count <> R.Count then
    Result := L.Count > R.Count
  else if L.A <> R.A then
    Result := L.A < R.A
  else
    Result := L.B < R.B;
end;

procedure HeapSwap(var X, Y: TPairHeapEntry);
var
  T: TPairHeapEntry;
begin
  T := X;
  X := Y;
  Y := T;
end;

procedure HeapPush(var Heap: TPairHeap; A, B, Count: Integer);
var
  I, Parent: Integer;
begin
  if Count <= 0 then Exit;

  if Heap.Count >= Length(Heap.Items) then begin
    if Length(Heap.Items) = 0 then
      SetLength(Heap.Items, 16)
    else
      SetLength(Heap.Items, Length(Heap.Items) * 2);
  end;

  I := Heap.Count;
  Heap.Items[I].A := A;
  Heap.Items[I].B := B;
  Heap.Items[I].Count := Count;
  Inc(Heap.Count);

  while I > 0 do begin
    Parent := (I - 1) div 2;
    if not HeapEntryGreater(Heap.Items[I], Heap.Items[Parent]) then Break;

    HeapSwap(Heap.Items[I], Heap.Items[Parent]);
    I := Parent;
  end;
end;

function HeapPop(var Heap: TPairHeap; out Entry: TPairHeapEntry): Boolean;
var
  I, Left, Right, Best: Integer;
begin
  if Heap.Count <= 0 then begin
    Result := False;
    Exit;
  end;

  Entry := Heap.Items[0];
  Dec(Heap.Count);

  if Heap.Count > 0 then begin
    Heap.Items[0] := Heap.Items[Heap.Count];

    I := 0;
    while True do begin
      Left := I * 2 + 1;
      Right := Left + 1;
      Best := I;

      if (Left < Heap.Count) and HeapEntryGreater(Heap.Items[Left], Heap.Items[Best]) then
        Best := Left;

      if (Right < Heap.Count) and HeapEntryGreater(Heap.Items[Right], Heap.Items[Best]) then
        Best := Right;

      if Best = I then
        Break;

      HeapSwap(Heap.Items[I], Heap.Items[Best]);
      I := Best;
    end;
  end;

  Result := True;
end;

procedure InitPairHeapFromHash(const H: TPairHash; var Heap: TPairHeap);
var
  I: Integer;
begin
  InitPairHeap(Heap, H.Used + 1024);

  for I := 0 to H.Capacity - 1 do
    if (H.Entries[I].State = psUsed) and (H.Entries[I].Count > 0) then
      HeapPush(Heap, H.Entries[I].A, H.Entries[I].B, H.Entries[I].Count);
end;

procedure PairIncHashHeap(var H: TPairHash; var Heap: TPairHeap; A, B: Integer);
var
  C: Integer;
begin
  if not PairAllowed(A, B) then Exit;

  C := PairIncHash(H, A, B);
  HeapPush(Heap, A, B, C);
end;

procedure PairDecHashHeap(var H: TPairHash; var Heap: TPairHeap; A, B: Integer);
var
  C: Integer;
begin
  if not PairAllowed(A, B) then Exit;

  C := PairDecHash(H, A, B);
  if C > 0 then
    HeapPush(Heap, A, B, C);
end;

function FindBestPairHeap(const H: TPairHash; var Heap: TPairHeap; out A, B: Integer): Integer;
var
  E: TPairHeapEntry;
  CurrentCount: Integer;
begin
  A := -1;
  B := -1;
  Result := 0;

  while HeapPop(Heap, E) do begin
    CurrentCount := PairGetCount(H, E.A, E.B);

    if CurrentCount <= 0 then Continue;

    if CurrentCount = E.Count then begin
      A := E.A;
      B := E.B;
      Result := CurrentCount;
      Exit;
    end;

    // Stale heap entry. Push the current count and keep looking.
    HeapPush(Heap, E.A, E.B, CurrentCount);
  end;
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
procedure UpdatePairsForMergeHash(Node: PTokenNode; NewTok: Integer; var H: TPairHash; var Heap: TPairHeap);
var
  A, B: Integer;
begin
  if (Node = nil) or (Node^.Next = nil) then Exit;

  A := Node^.Tok;
  B := Node^.Next^.Tok;

  // Remove (A, B).
  PairDecHashHeap(H, Heap, A, B);

  // Remove (Prev, A).
  if Node^.Prev <> nil then
    PairDecHashHeap(H, Heap, Node^.Prev^.Tok, A);

  // Remove (B, Next).
  if Node^.Next^.Next <> nil then
    PairDecHashHeap(H, Heap, B, Node^.Next^.Next^.Tok);

  // Add (Prev, NewTok).
  if Node^.Prev <> nil then
    PairIncHashHeap(H, Heap, Node^.Prev^.Tok, NewTok);

  // Add (NewTok, Next).
  if Node^.Next^.Next <> nil then
    PairIncHashHeap(H, Heap, NewTok, Node^.Next^.Next^.Tok);
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
procedure MergeAllPairsHash(var Head, Tail: PTokenNode; A, B, NewTok: Integer; var H: TPairHash; var Heap: TPairHeap);
var
  Cur: PTokenNode;
begin
  Cur := Head;

  while (Cur <> nil) and (Cur^.Next <> nil) do begin
    if not (IsSpecial(Cur^.Tok) or IsSpecial(Cur^.Next^.Tok)) then begin
      if (Cur^.Tok = A) and (Cur^.Next^.Tok = B) then begin
        UpdatePairsForMergeHash(Cur, NewTok, H, Heap);
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
  if NewTok >= Length(SymbolTable) then
    SetLength(SymbolTable, NewTok + 1);

  SymbolTable[NewTok] := SymbolTable[A] + SymbolTable[B];
end;

{ Apply the BPE encoder }
// Main training loop, traverse the merges.
procedure TrainBPEHash(var Head, Tail: PTokenNode; MaxMerges: Integer;
  MaxSymbols: Integer; var MergeCount, StartSymbol: Integer);
var
  m, BestCount, A, B: Integer;
  f, BaseName: string;
  BreakRequested: Boolean;
  H: TPairHash;
  Heap: TPairHeap;

  procedure ReadMergeIfKeyPressed;
  var
    key: Char;
  begin
    key := CheckForControlKey;
    case key of
      'x', 'X': begin
        Writeln('Exit requested. Stopping execution.');
        Pause;
        Halt;
      end;
      'b', 'B': begin
        Writeln('Break requested. Exiting loop.');
        BreakRequested := True;
      end;
      'v', 'V': begin
        VerboseTokenize := not VerboseTokenize;
        Writeln('Verbose tokenize mode: ', VerboseTokenize);
        Pause;
      end;
      'i', 'I': begin
        Writeln;
        ReportProgramInfo;
        Pause;
      end;
      'p', 'P':
        Pause;
      'm', 'M': begin
        Writeln;
        Writeln('Maximum symbols = ', MaxSymbols, '. Current symbols = ', Length(SymbolTable),
          '. Maximum merges = ', MaxMerges, '. Hash capacity = ', H.Capacity, '. Used slots = ', H.Used, '. Heap entries = ', Heap.Count, '. Best count = ', BestCount, '.');
        Write(DateTimeToStr(Now), '  X = Exit program. B = Break out of merge loop. V = toggle Verbose mode. I = program Information. ');
        Writeln('P = Pause. M = Merging information. S = Save. Symbolizing and merging...');
        Pause;
      end;
      's', 'S': begin
        try
          if Trim(WorkingName) = '' then
            BaseName := 'symboltable'
          else
            BaseName := ChangeFileExt(ExtractFileName(WorkingName), '');

          if Trim(BaseName) = '' then
            BaseName := 'symboltable';

          if Trim(SymbolDir) = '' then begin // Symboldir now seems to work.
            SymbolDir := IncludeTrailingPathDelimiter(GetCurrentDir) +
              'WesChatWork' + DirectorySeparator + 'symbols' + DirectorySeparator;
            ForceDirectories(SymbolDir);
          end;

          // Make sure the directory name is clean.
          SymbolDir := IncludeTrailingPathDelimiter(SymbolDir);

          if not DirectoryExists(SymbolDir) then begin
            Writeln('Creating symbol directory: ', SymbolDir);
            ForceDirectories(SymbolDir);
          end;

          f := SymbolDir + BaseName + '_' + FormatDateTime('yyyy-mm-dd_hhnnss', Now) + '.sym';

          Writeln('Saving symbol table to: ', f);

          SaveSymbolTable(f, SymbolTable);

          Pause;
        except
          on E: Exception do begin
            Writeln('Error saving symbol table: ', E.ClassName, ' ', E.Message);
            Writeln('SymbolDir = "', SymbolDir, '"');
            Writeln('WorkingName = "', WorkingName, '"');
            Writeln('BaseName = "', BaseName, '"');
            Writeln('Target file = "', f, '"');
            Pause;
          end;
        end;
      end;
    end;
  end;

begin
  MergeCount := 0;
  BreakRequested := False;

  Write(DateTimeToStr(Now), '  X = Exit program. B = Break out of merge loop. V = toggle Verbose mode. I = program Information. ');
  Writeln('P = Pause. M = Merging information. S = Save. Symbolizing and merging...');
  Writeln;

  if DisplayMergeWork then
    Writeln('--- List of Merges (Hash) ---');

  // Build pair counts once, then maintain them incrementally.
  InitPairHash(H, MaxPairCount * 2 + 1024);
  InitPairHashFromList(Head, H);
  InitPairHeapFromHash(H, Heap);

  // Merge loop.
  for m := 1 to MaxMerges do begin
    // Check for key pressed.
    if PauseIfKeyPressed then
      ReadMergeIfKeyPressed;

    // Check for break request.
    if BreakRequested then Break;

    // Stop if symbol table got too large.
    if Length(SymbolTable) >= MaxSymbols then begin
      Writeln('Stopping: symbol table reached ', MaxSymbols, ' entries.');
      Break;
    end;

    // The open-address hash table never truly deletes zero-count pairs.
    // If it gets crowded, rebuild it from the current linked list.
    if H.Used > (H.Capacity * 7) div 10 then begin
      InitPairHash(H, H.Capacity * 2);
      InitPairHashFromList(Head, H);
      InitPairHeapFromHash(H, Heap);
    end;

    BestCount := FindBestPairHeap(H, Heap, A, B);

    // Stop if no useful merges remain.
    if BestCount < 2 then begin
      Writeln;
      Writeln('Stopping: no more valid merges at iteration ', m, '.');
      Break;
    end;

    // Perform merge.
    MergeAllPairsHash(Head, Tail, A, B, StartSymbol, H, Heap);

    // Lazy heap entries accumulate. Compact occasionally by rebuilding
    // the heap from the current hash counts.
    if Heap.Count > (H.Used * 8 + 100000) then
      InitPairHeapFromHash(H, Heap);

    AddMergeSymbol(StartSymbol, A, B);
    RecordMerge(Merges, MergeCount, A, B, StartSymbol);

    Inc(MergeCount);
    Inc(StartSymbol);

    if DisplayMergeWork then begin
      Write(MergeCount, ' Merged (', A:5, ',', B:5, ') -> (', StartSymbol - 1:5, ') #', BestCount);
      if (MergeCount mod 4) = 0 then
        Writeln
      else
        Write('  |  ');
    end;
  end;

  Writeln('Hash tokenization complete. Total merges: ', MergeCount, '.');
  // Pause;
end;

{ Computations and reports }
// Calculate time statistics.
procedure CalculateTimeStatistics;
begin
  // Total elapsed time.
  ElapsedMS := MilliSecondsBetween(t0, t1) - Round(StopTime * 86400000.0);
  Hours := ElapsedMS div 3600000;
  Mins := (ElapsedMS mod 3600000) div 60000;
  Secs := (ElapsedMS mod 60000) / 1000.0;
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

  Writeln('--- Symbols Statistics ---');
  Writeln('Number of raw byte symbols: ', 256);
  Writeln('Number of special symbols: ', 4);
  Writeln('Number of merged symbols: ', nSymbols - 260);

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

  // Median: Sort the Lengths array.
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
end;

// Count token nodes.
function CountTokenNodes(Head: PTokenNode): Integer;
var
  Cur: PTokenNode;
begin
  Result := 0;
  Cur := Head;

  while Cur <> nil do begin
    Inc(Result);
    Cur := Cur^.Next;
  end;
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
    Writeln('Longest symbol:');
    Writeln('  Index: ', maxIndex);
    Writeln('  Length: ', maxLen);
    Writeln('  Value: "', SymbolTable[maxIndex], '"');
  end;
end;

{ Report Statistics }
// Report basic statistics (time, file names).
procedure ReportBasicStatistics;
var
  i: Integer;
begin
  Writeln;
  Writeln('--- File Information ---');
  Writeln('Files used in symbol table: ');
  for i := 0 to High(CorpusFileNames) do
    Writeln(CorpusFileNames[i], '  ');
  Writeln;

  Writeln('--- Token Statistics ---');
  Writeln('Final BPE tokens: ', FinalTokenCount);
  Writeln('Bytes per token: ', nCorpus / FinalTokenCount:0:4);
  Writeln('Token count / original bytes: ', FinalTokenCount / nCorpus:0:4);

  Writeln('--- Time Statistics ---');
  Writeln('Start time: ', DateTimetoStr(t0), '     End time: ', DateTimeToStr(t1));
  Writeln('Total elapsed time: ', Hours, ' hours, ', Mins, ' min ', Secs: 4: 4, ' sec');
  Writeln('Number of symbols: ', nSymbols);
  Writeln('Original text size (bytes/tokens): ', nCorpus);
  if not FromSymbolTable then begin
    Writeln('Tokens per second (total): ', nCorpus / (ElapsedMS / 1000): 6: 4);
    Writeln;
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
// Save metadata/report statistics to a .sym.tok file.
// Uses full path handling. Does not change current directory.
procedure SaveMetaData(const MetaFileName: String);
var
  SaveOut: Text;
  OutName, OutDir: String;
  Redirected: Boolean;
begin
  OutName := Trim(MetaFileName);

  // If caller passes blank filename, build a default.
  if OutName = '' then begin
    if Trim(WorkingName) <> '' then
      OutName := ChangeFileExt(ExtractFileName(WorkingName), '') + '.sym.tok'
    else
      OutName := 'symbolize.sym.tok';
  end;

  // If caller passed no path, save into LogDir if available.
  if ExtractFilePath(OutName) = '' then begin
    if Trim(LogDir) <> '' then
      OutName := IncludeTrailingPathDelimiter(LogDir) + OutName
    else
      OutName := IncludeTrailingPathDelimiter(GetCurrentDir) + OutName;
  end;

  // Add .sym.tok extension if missing.
  if ExtractFileExt(OutName) = '' then
    OutName := OutName + '.sym.tok';

  OutDir := ExtractFilePath(OutName);

  try
    // Create destination folder, but never call ForceDirectories('').
    if Trim(OutDir) <> '' then
      ForceDirectories(OutDir);

    // Save current console Output.
    SaveOut := Output;
    Redirected := False;

    try
      // Redirect Output to metadata file.
      Assign(Output, OutName);
      Rewrite(Output);
      Redirected := True;

      ReportStatistics;

    finally
      // Always restore console Output.
      if Redirected then
        Close(Output);

      Output := SaveOut;
    end;

    Writeln('File ', OutName, ' successfully saved.');
    Writeln;

  except
    on E: Exception do begin
      // Try to restore Output even if something failed early.
      Output := SaveOut;

      Writeln('Error saving metadata: ', E.ClassName, ' ', E.Message);
      Writeln('Target metadata file = "', OutName, '"');
      Writeln;
    end;
  end;
end;

// Save merge table.
// Uses full path handling. Does not change current directory.
procedure SaveMergeTable(const Merges: TMergeArray; MergeFileName: String);
var
  F: file;
  i, n: Integer;
  OutName, OutDir: String;
  FileOpen: Boolean;
begin
  OutName := Trim(MergeFileName);

  // If caller passes blank filename, build a default.
  if OutName = '' then begin
    if Trim(WorkingName) <> '' then
      OutName := WorkingName + '.mer'
    else
      OutName := 'symbolize.mer';
  end;

  // If caller passed no path, save into SymbolDir if available.
  // If you later add MergeDir, use MergeDir here instead.
  if ExtractFilePath(OutName) = '' then begin
    if Trim(MergeDir) <> '' then
      OutName := IncludeTrailingPathDelimiter(MergeDir) + OutName
    else if Trim(SymbolDir) <> '' then
      OutName := IncludeTrailingPathDelimiter(SymbolDir) + OutName
    else
      OutName := IncludeTrailingPathDelimiter(GetCurrentDir) + OutName;
  end;

  // Add .mer extension if missing.
  if ExtractFileExt(OutName) = '' then
    OutName := OutName + '.mer';

  OutDir := ExtractFilePath(OutName);
  FileOpen := False;

  try
    // Create destination folder, but never call ForceDirectories('').
    if Trim(OutDir) <> '' then
      ForceDirectories(OutDir);

    Assign(F, OutName);
    Rewrite(F, 1);
    FileOpen := True;

    n := Length(Merges);
    BlockWrite(F, n, SizeOf(n));

    for i := 0 to n - 1 do begin
      BlockWrite(F, Merges[i].A, SizeOf(Integer));
      BlockWrite(F, Merges[i].B, SizeOf(Integer));
      BlockWrite(F, Merges[i].NewSym, SizeOf(Integer));
    end;

    Close(F);
    FileOpen := False;

    Writeln('File ', OutName, ' successfully saved.');

  except
    on E: Exception do begin
      if FileOpen then
        Close(F);

      Writeln('Error saving merge table: ', E.ClassName, ' ', E.Message);
      Writeln('Target merge file = "', OutName, '"');
    end;
  end;
end;

// Run the tokenizer.
procedure RunSymbolize(const Corpus: TBVector);
begin
  // Reset for new run.
  MergeCount := 0;
  SetLength(Merges, 0);

  // Timing.
  t0 := Now;       // Start of timing for entire tokenization;
  StopTime := 0;   // Time to subtract from timing.

  // Initialize base byte symbols plus special tokens.
  InitSymbolTable;
  StartSymbol := Length(SymbolTable);
  nSymbols := Length(SymbolTable);

  // Convert byte corpus to initial token list.
  BuildTokenListFromCorpus(Corpus);
  nCorpus := Length(Corpus);

  Writeln('Symbolizing and merging started.');
  Writeln('Maximum symbols = ', MaxSymbols, '. Base symbols = ', nSymbols, '. Maximum merges = ', MaxMerges, '. Maximum pair counts = ', MaxPairCount, '.');

  TrainBPEHash(Head, Tail, MaxMerges, MaxSymbols, MergeCount, StartSymbol);

  nSymbols := Length(SymbolTable);
  nVocab := nSymbols;
  FinalTokenCount := CountTokenNodes(Head);

  // Timing.
  t1 := Now;

  //nSymbols := Length(SymbolTable);
  // Display symbol table.
  if VerboseTokenize then
    DisplayByteSymbolTable(SymbolTable);

  // Report statistics.
  if VerboseTokenize then
    ReportStatistics;

  FreeTokenList(Head, Tail);
  Writeln('Symbolizing and merging ended.');
end;

end.
