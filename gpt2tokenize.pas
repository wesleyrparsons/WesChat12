unit GPT2Tokenize;

{$mode objfpc}{$H+}{$I proprietary.txt}

{ WesChat, Version 1.2, begun January 10, 2026, by Wesley R. Parsons, wespar@bellouth.net, www.wesparsons.com.}

interface

uses
  Classes,
  Display,
  Fpjson,
  Global,
  IOHandler,
  Jsonparser,
  Math,
  SysUtils,
  WesTokenize;

procedure RunGPT2Tokenize(const FileName: string; var TokenizedCorpus: TIVector);
procedure EnsureGPT2VocabLoaded;
procedure LoadVocab(const FileName: string; Vocab: TStringList);
function DisplayToken(const S: UnicodeString): AnsiString;

implementation

type
  TRBStringArray = array of RawByteString;
  TUStringArray = array of UnicodeString;

var
  InputBytes: RawByteString;
  EncodedText: UnicodeString;

// Ensure that vocab1.json is loaded.
procedure EnsureGPT2VocabLoaded;
begin
  if Vocab = nil then
    Vocab := TStringList.Create;

  if Vocab.Count = 0 then
    LoadVocab('vocab1.json', Vocab);
end;

// Load a file of bytes.
function LoadFileRaw(const FileName: string): RawByteString;
var
  FS: TFileStream;
  i: Integer;
begin
  FS := TFileStream.Create(FileName, fmOpenRead or fmShareDenyNone);
  Result := '';
  try
    SetLength(Result, FS.Size);
    if FS.Size > 0 then
      FS.ReadBuffer(Result[1], FS.Size);
  finally
    FS.Free;
  end;

 if VeryVerboseTokenize then begin
    Writeln('Display rawbytes read:');
    for i := 500 to high(result) do
      Writeln(i: 5, ord(Result[i]): 5, '   *', Result[i], '* ');
    Readln;
  end;
end;

// Encode bytes to ChatGPT2 unicode.
function GPT2ByteToUnicode(B: Byte): WideChar;
var
  i, n: Integer;
begin
  if ((B >= 33) and (B <= 126)) or
     ((B >= 161) and (B <= 172)) or
      (B >= 174) then begin
    Result := WideChar(B);
    Exit;
  end;

  n := 0;
  for i := 0 to B - 1 do
    if not (((i >= 33) and (i <= 126)) or
            ((i >= 161) and (i <= 172)) or
            ((i >= 174) and (i <= 255))) then
      Inc(n);

  Result := WideChar(256 + n);
end;

function EncodeBytesToUnicode(const x: RawByteString): UnicodeString;
var
  i: Integer;
begin
  SetLength(Result, Length(x));

  for i := 1 to Length(x) do
    Result[i] := GPT2ByteToUnicode(Byte(x[i]));
end;

function DisplayToken(const S: UnicodeString): AnsiString;
var
  i, cp: Integer;
  b: Byte;
begin
  Result := '';

  for i := 1 to Length(S) do begin
    cp := Ord(S[i]);

    // ASCII printable.
    if (cp >= 32) and (cp <= 126) then
      Result := Result + AnsiChar(cp)

    // GPT2 fallback range.
    else if (cp >= $0100) and (cp <= $01FF) then begin
      b := cp - $0100;

    // One rule for control bytes 0..32
      if b <= 32 then
        Result := Result + Chr(b)
      else
        Result := Result + '?';

    end;
  end;
end;

// New NextToken version, 3/9/2026.
function IsWordChar(ch: WideChar): Boolean;
begin
  Result :=
    ((ch >= 'A') and (ch <= 'Z')) or
    ((ch >= 'a') and (ch <= 'z')) or
    ((ch >= '0') and (ch <= '9'));
end;

function IsPunct(ch: WideChar): Boolean;
begin
  Result := not IsWordChar(ch) and (ch <> ' ') and (ch <> #$0120);
end;

function NextToken(const S: UnicodeString; var idx: Integer): UnicodeString;
var
  start: Integer;
  ch: WideChar;
begin
  Result := '';
  if idx > Length(S) then Exit;

  ch := S[idx];

  { 1. word with leading space marker (Ġ) }
  if ch = #$0120 then begin
    start := idx;
    Inc(idx);

    while (idx <= Length(S)) and IsWordChar(S[idx]) do
      Inc(idx);

    Result := Copy(S, start, idx - start);
    Exit;
  end;

  { 2. plain word }
  if IsWordChar(ch) then begin
    start := idx;

    while (idx <= Length(S)) and IsWordChar(S[idx]) do
      Inc(idx);

    Result := Copy(S, start, idx - start);
    Exit;
  end;

  { 3. punctuation }
  if IsPunct(ch) then begin
    start := idx;

    while (idx <= Length(S)) and IsPunct(S[idx]) do
      Inc(idx);

    Result := Copy(S, start, idx - start);
    Exit;
  end;

  { 4. skip spaces }
  Inc(idx);
end;

{Old version of NextToken.
function IsLetterOrDigit(c: Widechar): Boolean;
begin
  Result :=
    ((c >= 'a') and (c <= 'z')) or
    ((c >= 'A') and (c <= 'Z')) or
    ((c >= '0') and (c <= '9')) or
    (c > #255);  // remapped
end;

function IsWhitespace(c: Widechar): Boolean;
begin
  Result :=
    (c = ' ') or   // space
    (c = #9)  or   // tab
    (c = #10) or   // LF
    (c = #13);     // CR
end;

function NextToken(const S: UnicodeString; var idx: Integer): UnicodeString;
var
  Start: Integer;
  ch: WideChar;
begin
  Result := '';

  while idx <= Length(S) do begin
    ch := S[idx];

    // 1. GPT‑2 byte‑fallback marker (U+0100..U+01FF) → start of a word token.
    if (Ord(ch) >= $0100) and (Ord(ch) <= $01FF) then begin
      start := idx;
      Inc(idx);

      // Consume ASCII letters/digits.
      while (idx <= Length(S)) and (S[idx] in ['A'..'Z','a'..'z','0'..'9','''', '"']) do
        Inc(idx);

      Result := Copy(S, Start, idx - Start);
      Exit;
    end;

    // 2. Punctuation → its own token.
    if ch in ['.', ',', ';', ':', '!', '?', '(', ')', '[', ']', '{', '}', '"', ''''] then begin
      Result := ch;
      Inc(idx);
      Exit;
    end;

    // 3. Otherwise skip (spaces, other unicode, etc.).
    Inc(idx);
  end;
end;}

{ No longer used.
procedure GPTChunk(const Text: UnicodeString; var Chunks: TUStringArray);
var
  i, Start, Count: Integer;
  Mode: (mNone, mWord, mSpace, mPunct);
  c: Widechar;

  procedure AddChunk(aStart, aLen: Integer);
  begin
    if aLen <= 0 then Exit;
    Inc(count);
    SetLength(Chunks, count);
    Chunks[Count - 1] := Copy(Text, aStart, aLen);
  end;

begin
  SetLength(Chunks, 0);
  Count := 0;
  Mode := mNone;
  Start := 1;

  for i := 1 to Length(Text) do begin
    c := Text[i];

    if IsLetterOrDigit(c) then begin
      if Mode <> mWord then begin
        AddChunk(Start, i - Start);
        Start := i;
        Mode := mWord;
      end;
    end
    else if IsWhitespace(c) then begin
      if Mode <> mSpace then begin
        AddChunk(Start, i - Start);
        Start := i;
        Mode := mSpace;
      end;
    end
    else begin
      if mode <> mPunct then begin
        AddChunk(Start, i - Start);
        Start := i;
        Mode := mPunct;
      end;
    end;
  end;

  AddChunk(start, Length(Text) - start + 1);

  Writeln('Chunks (first 10) Length =', Length(Chunks));
  for i := 0 to 9 {High(Chunks)} do begin
    Writeln(i, ' *', Chunks[i], '*  Raw: ');
    for j := 1 to Length(Chunks[i]) do
      Write(' @', ord(Chunks[i][j]), '@   ');
    Readln;
  end;
end;}

procedure DisplayVocab(const a, b: Integer);
var
  i, j: Integer;
  s: string;
begin
 Writeln('Vocab ', a, ' to ', b);
  for i := a to b do begin
    Write(i, ' ', 'Vocab[i]: ', Vocab[i], ' ');
    Write('Raw bytes: ');
    s := Vocab[i];
    for j := 1 to Length(s) do
      Write(Ord(s[j]), ' ');
    Writeln;
  end;
end;

// Load vocab1.json.
procedure LoadVocab(const FileName: string; Vocab: TStringList);
var
  Raw: RawByteString;
  JSON: TJSONData;
  Obj: TJSONObject;
  FS: TFileStream;
  i, TokenID, MaxTokenID: Integer;
begin
  Vocab.Clear;
  Vocab.Sorted := False;
  Vocab.Duplicates := dupAccept;
  Vocab.CaseSensitive := True;
  Vocab.OwnsObjects := False;

  if not FileExists(FileName) then begin
    Writeln('ERROR: Vocabulary file not found: ', FileName);
    Exit;
  end;

  FS := TFileStream.Create(FileName, fmOpenRead or fmShareDenyNone);
  try
    SetLength(Raw, FS.Size);

    if FS.Size > 0 then
      FS.ReadBuffer(Raw[1], FS.Size);
  finally
    FS.Free;
  end;

  JSON := GetJSON(Raw);
  try
    if not (JSON is TJSONObject) then begin
      Writeln('Invalid vocabulary JSON.');
      Exit;
    end;

    Obj := TJSONObject(JSON);

    // Find the largest token ID.
    MaxTokenID := -1;

    for i := 0 to Obj.Count - 1 do begin
      TokenID := Obj.Items[i].AsInteger;

      if TokenID > MaxTokenID then
        MaxTokenID := TokenID;
    end;

    // Create one list position for every token ID.
    Vocab.Capacity := MaxTokenID + 1;

    for i := 0 to MaxTokenID do
      Vocab.Add('');

    // Place each token at its actual ID.
    for i := 0 to Obj.Count - 1 do begin
      TokenID := Obj.Items[i].AsInteger;

      if (TokenID >= 0) and (TokenID <= MaxTokenID) then
        Vocab[TokenID] := Obj.Names[i];
    end;

  finally
    JSON.Free;
  end;

  Writeln('End of loading vocabulary. Length of Vocab: ', Vocab.Count);
end;

{procedure LoadVocab(const FileName: string; Vocab: TStringList);
var
  Raw: RawByteString;
  JSON: TJSONData;
  Obj: TJSONObject;
  i: Integer;
  FS: TFileStream;
begin
  Vocab.Clear;
  Vocab.Sorted := False;
  Vocab.Duplicates := dupIgnore;
  Vocab.CaseSensitive := True;
  Vocab.OwnsObjects := False;

  if not FileExists(FileName) then begin
    Writeln('ERROR: File not found.');
    Exit;
  end;

  FS := TFileStream.Create(FileName, fmOpenRead or fmShareDenyNone);
  try
    SetLength(Raw, FS.Size);
    if FS.Size > 0 then
      FS.ReadBuffer(Raw[1], FS.Size);
  finally
    FS.Free;
  end;

  JSON := GetJSON(Raw);   // Pass RAW bytes.
  try
    if not (JSON is TJSONObject) then begin
      Writeln('Invalid vocab.json.');
      Halt;
    end;

    Obj := JSON as TJSONObject;
    // Writeln('Size of JSON symbol table: ', Obj.count);

    for i := 0 to Obj.Count - 1 do
      Vocab.AddObject(Obj.Names[i], TObject(PtrInt(Obj.Items[i].AsInteger)));
  finally
    JSON.Free;
  end;

  WriteLn('End of loading symbol table. Length of vocab: ', Vocab.Count);
  if VeryVerboseTokenize then begin
    DisplayVocab(0, 9);
    DisplayVocab(120, 130);
    DisplayVocab(288, 301);
    if VeryVerboseTokenize then Pause;
  end;
end;}

procedure LoadMerges(const FileName: string; Merges: TStringList);
var
  SL: TStringList;
  Line: string;
  Parts: TRBStringArray;
  i: Integer;
begin
  Merges.Clear;
  Merges.CaseSensitive := True;
  Merges.Sorted := True;     // Changed from False to speed up.
  Merges.Duplicates := dupIgnore;

  SL := TStringList.Create;
  try
    SL.LoadFromFile(FileName, TEncoding.UTF8);

    for i := 0 to SL.Count - 1 do begin
      Line := Trim(SL[i]);
      if (Line = '') or (Line[1] = '#') then Continue;

      Parts := Line.Split([' ']);
      // Rank = insertion order.
      if Length(Parts) = 2 then
        Merges.AddObject(Parts[0] + ' ' + Parts[1], TObject(PtrInt(Merges.Count)));
    end;

  finally
    SL.Free;
  end;

  Writeln('End of loading merges. Length of merges: ', Merges.Count);
  if VeryVerboseTokenize then begin
    Write('Merges 0: ',Merges[0],'   Raw: ');
    for i := 1 to Length(Merges[0]) do
      Write(ord(merges[0][i]), ' ');
    Writeln;
    Write('Merges 1: ', Merges[1],'   Raw: ');
    for i := 1 to Length(Merges[1]) do
      Write(ord(merges[1][i]), ' ');
    Writeln;
    Write('Merges 2: ', Merges[2],'   Raw: ');
    for i := 1 to Length(Merges[2]) do
      Write(ord(merges[2][i]), ' ');
    Writeln;
  end;
end;

{This version returns words split at the spce as Words[_].
The tokenizer runs faster.}
{procedure PreTokenize(const Text: String; var Words: TStringArray);
var
  Parts: TStringArray;
  i: Integer;
const
  GPTSpace: AnsiString = #$C4#$A0;  { UTF-8 for Ġ }
begin
 Parts := Text.Split([' ']);
 SetLength(Words, Length(Parts));

  for i := 0 to High(Parts) do
    if i = 0 then
      Words[i] := Parts[i]
    else  begin
      Words[i] := GPTSpace + Parts[i];
    end;
end;}

{This version returns a complete line as Words[0].
The tokenizer runs much slower.}
{procedure PreTokenize(const Text: String; var Words: TStringArray);
var
  Parts: TStringArray;
  i, j, k: Integer;
const
  GPTSpace: AnsiString = #$C4#$A0;  { UTF-8 for Ġ }
 begin
   Writeln('start pre ', Text);

   setlength(words, 1);
   Words[0] := '';
   for i := 1 to High(Text) do
     if Text[i] = ' ' then
       Words[0] := Words[0] + GPTSpace
     else
       Words[0] := Words[0] + Text[i];

   Writeln('Pretoken result: ', words[0], ' Raw: ');
   for k := 1 to High(Words[0]) do
     Write(Words[0][k]);
end;}

function UTF8CharLen(P: PChar): Integer;
var
  B: Byte;
begin
  B := Byte(P^);
  if B < $80 then Result := 1
  else if (B and $E0) = $C0 then Result := 2
  else if (B and $F0) = $E0 then Result := 3
  else if (B and $F8) = $F0 then Result := 4
  else Result := 1;
end;

procedure BPE(const Word: string; const Merges: TStringList; var Output: TStringArray);
var
  Symbols: TStringArray;
  Pair, BestPair: string;
  i, j, Rank, BestRank: Integer;
  Found: Boolean;
var
  p, len: Integer;
begin
  { 1. Initialize symbols as UTF8 bytes }
  begin
    p := 1;
    SetLength(Symbols, 0);
    while p <= Length(Word) do begin
      len := UTF8CharLen(@Word[p]);
      SetLength(Symbols, Length(Symbols)+1);
      Symbols[High(Symbols)] := Copy(Word, p, len);
      Inc(p, len);
    end;
  end;

  { 2. Main merge loop }
  while True do begin
    BestRank := MaxInt;
    BestPair := '';
    Found := False;

    { 2a. Find the best-ranked mergeable pair }
    for i := 0 to High(Symbols)-1 do begin
      Pair := Symbols[i] + ' ' + Symbols[i+1];   { GPT2 uses space delimiter }

      j := Merges.IndexOf(Pair);
      if j >= 0 then begin
        Rank := PtrInt(Merges.Objects[j]);
        if Rank < BestRank then begin
          BestRank := Rank;
          BestPair := Pair;
          Found := True;
        end;
      end;
    end;

    if not Found then Break;

    { 2b. Apply the best merge everywhere }
    i := 0;
    while i < High(Symbols) do begin
      Pair := Symbols[i] + ' ' + Symbols[i + 1];
      if Pair = BestPair then begin
        Symbols[i] := Symbols[i] + Symbols[i + 1];  { merge into UTF‑8 string }

        { shift left }
        for j := i+1 to High(Symbols) - 1 do
          Symbols[j] := Symbols[j + 1];

        SetLength(Symbols, Length(Symbols) - 1);
      end
      else
        Inc(i);
    end;
  end;

  Output := Symbols;
end;

procedure ShowRaw(const x: string);
var
  j: Integer;
begin
  Write('Raw: ');
  for j := 1 to Length(x) do
    Write(ord(x[j]), ' ');
  Writeln;
end;

procedure TokenizeFile(const Corpus: String; const Vocab, Merges: TStringList;
  var TokenIDs: TIVector);
var
  Words: TUStringArray;
  Pieces: TStringArray;
  tok: UnicodeString;
  i, j, k, idx, Count, iWord: Integer;
begin
  Count := 0;
  SetLength(TokenIDs, 0);

  InputBytes := LoadFileRaw(Corpus);  // This is 1-based.
  EncodedText := EncodeBytesToUnicode(InputBytes); // This is 1-based.

  // Because NextToken needs a leading $0120.
  EncodedText := WideChar($0120) + EncodedText;

  i := 1;
  iWord := 0;
  while True do begin
    tok := NextToken(EncodedText, i);
    if tok = '' then Break;
    SetLength(Words, iWord + 1);
    // Writeln('iword ', iword, ' tok', tok); Readln;
    Words[iWord] := Tok;
    Inc(iWord);
  end;

  // Remove the $0120 that was added.
  if (Length(Words) > 0) and (Length(Words[0]) > 0) then
    Words[0] := Copy(Words[0], 2, Length(Words[0]) - 1);

  // Byte-pair encoding.
  for j := 0 to High(Words) do begin
    BPE(UTF8Encode(Words[j]), Merges, Pieces);

    for k := 0 to High(Pieces) do begin
      idx := Vocab.IndexOf(Pieces[k]);

      if idx >= 0 then begin
        Inc(Count);
        SetLength(TokenIDs, Count);
        TokenIDs[Count - 1] := idx;
      end;

      {idx := Vocab.IndexOf(Pieces[k]);
      if idx >= 0 then begin
        Inc(Count);
        SetLength(TokenIDs, Count);
        TokenIDs[Count - 1] := PtrInt(Vocab.Objects[idx]);
      end;}
    end;
  end;
end;

function DecodeToken(const s: UTF8String): UnicodeString;
var
  u: UnicodeString;
  i: Integer;
  c: Word;
begin
  u := UTF8Decode(s);
  Result := '';

  for i := 1 to Length(u) do begin
    c := Ord(u[i]);

    case c of
      $0120: Result := Result + WideChar($0020);  // Ġ -> space
      $010D: Result := Result + WideChar($000D);  // č -> CR
      $010A: Result := Result + WideChar($000A);  // Ċ -> LF
      $0009: Result := Result + WideChar($0009);  // tab
    else
      if c < 128 then
        Result := Result + WideChar(c)
      else if (c >= $0100) and (c <= $01FF) then
        Result := Result + WideChar('?')
      else
        Result := Result + u[i];
    end;
  end;
end;

// Main work flow.
procedure RunGPT2Tokenize(const FileName: string; var TokenizedCorpus: TIVector);
var
  Merges: TStringList;
  Tokens: TIVector;
  i, j: Integer;
  s: string;
  SaveOut: Text;
begin
  EnsureGPT2VocabLoaded;
  // Vocab := TStringList.Create;
  // LoadVocab('vocab1.json', Vocab);

  Merges := TStringList.Create;
  LoadMerges('merges.txt', Merges);

  if VeryVerboseTokenize then Pause;
  t0 := Now;
  Write('Tokenizing started.');
  TokenizeFile(FileName, Vocab, Merges, Tokens);
  t1 := Now;

  Writeln('End of three routines in tokenizing.');

  SetLength(TokenizedCorpus, Length(Tokens));

  nTokenizedCorpus := Length(TokenizedCorpus);
  for i := 0 to High(Tokens) do
    TokenizedCorpus[i] := Tokens[i];

  Writeln('After 3 routines, 100 tokens.');
  for i := 0 to Min(99, High(Tokens)) do begin
    Write(' *', i, ' ', Tokens[i], ' ', Vocab[Tokens[i]], '*  ');
    s := Vocab[Tokens[i]];
    Write('=');
    for j := 1 to Length(s) do
      Write(Ord(s[j]), ' ');
    Write('=');
  end;
  Writeln;
  if VeryVerboseTokenize then Pause;

  Writeln('Tokenizing ended.');
  if DisplayTokenVerification then begin
    Writeln('TokenizedCorpus: ');
    for i := 0 to High(TokenizedCorpus) do
      Write(TokenizedCorpus[i], ' ');
    Writeln;
  end;
  if VeryVerboseTokenize then Pause;

  // Report statistics.
  {if VerboseTokenize then
    ReportGPT2Statistics(TokenizedCorpus);}

  // Save TokenizedCorpus and other data.
  if SaveFiles and SaveTokenizationFiles then begin
    // Save token list.
    SaveTokenList(TokenizedCorpus, TokenDir + WorkingName + '.tok');

    // Save current Output.
    SaveOut := Output;

    // Redirect Output to log file.
    Assign(Output, LogDir + WorkingName + '.log');

    if FileExists(LogDir + WorkingName + '.log') then
      Append(Output)
    else
      Rewrite(Output);

    // ReportGPT2Statistics(TokenizedCorpus);

    // Restore Output to console.
    Close(Output);
    Output := SaveOut;
  end;

  // Need to Write out TC with j loop to deal with chr183.
  Readln;
  Writeln('Vocabulary for TokenizedCorpus:', High(TokenizedCorpus));

  for i := 0 to High(TokenizedCorpus) do begin
    s := Vocab[TokenizedCorpus[i]];
    Write(DecodeToken(s));
  end;

{  for i := 0 to High(TokenizedCorpus) do begin
    s := Vocab[TokenizedCorpus[i]];
    // Replace the GPT-2 "Ġ" marker with a space.
    if s = 'Ġ' then
      Write(' ')
    else begin
      for j := 1 to Length(s) do
        Write(s[j]);
    end;
  end;}
  Writeln;
  if VeryVerboseTokenize then Pause;

  if DisplayCorpusVerification then Begin
    Write('First ', DisplayLength, ' tokens of detokenized corpus: ');
    for i := 0 to DisplayLength do
      Write(DisplayToken(UTF8Decode(Vocab[TokenizedCorpus[i]])));
    Writeln;
    Pause;
  end;

  Merges.Free;
end;

end.

