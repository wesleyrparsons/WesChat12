unit gpt2_bpe;

{$mode objfpc}{$H+}

interface

uses
Classes, global, SysUtils, fpjson, jsonparser, fgl;

type
TStringIntMap = specialize TFPGMap<string, integer>;

TPair = record
A, B: string;
end;

TPairList = array of TPair;

TPairRankMap = specialize TFPGMap<string, integer>;

function LoadVocab(const FileName: string): TStringIntMap; function LoadMerges(const FileName: string): TPairList; function BuildPairRanks(const M: TPairList): TPairRankMap;

function BPE(const Token: string; PairRanks: TPairRankMap): string;
function Encode(const FileName: string; Vocab: TStringIntMap; PairRanks: TPairRankMap): TIVector;
//function Encode(const Text: string; Vocab: TStringIntMap; PairRanks: TPairRankMap): TIVector;
procedure runbpe;

implementation

function ReadFileToString(const FileName: string): string; var
FS: TFileStream;
S: TStringStream;
begin
FS := TFileStream.Create(FileName, fmOpenRead);
S := TStringStream.Create('');
S.CopyFrom(FS, FS.Size);
Result := S.DataString;
S.Free;
FS.Free;
end;

function LoadVocab(const FileName: string): TStringIntMap; var F: TFileStream; Parser: TJSONParser; J: TJSONData; Obj: TJSONObject; I: integer; begin Result := TStringIntMap.Create;

F := TFileStream.Create(FileName, fmOpenRead); Parser := TJSONParser.Create(F); J := Parser.Parse; Parser.Free; F.Free;

Obj := TJSONObject(J);

for I := 0 to Obj.Count - 1 do Result.Add(Obj.Names[I], Obj.Items[I].AsInteger);

Result.Sorted := True; J.Free; end;

function LoadMerges(const FileName: string): TPairList; var
SL: TStringList;
i, c: integer;
parts: TStringArray;
begin
SL := TStringList.Create;
SL.LoadFromFile(FileName);

c := SL.Count - 1; if c < 1 then
begin
SetLength(Result, 0);
SL.Free;
Exit;
end;

SetLength(Result, c); for i := 1 to SL.Count - 1 do
begin
parts := SL[i].Split(' '); Result[i - 1].A := parts[0]; Result[i - 1].B := parts[1]; end;

SL.Free; end;

function BuildPairRanks(const M: TPairList): TPairRankMap; var
i: integer;
key: string;
begin
Result := TPairRankMap.Create; for i := 0 to High(M) do
begin
key := M[i].A + ' ' + M[i].B; Result.Add(key, i); end;

Result.Sorted := True; end;

function GetPairs(const S: TStringArray): TStringArray; var
i: integer;
begin
SetLength(Result, 0); if Length(S) < 2 then Exit;

SetLength(Result, Length(S)-1); for i := 0 to High(S) - 1 do
Result[i] := S[i] + ' ' + S[i+1]; end;

function BPE(const Token: string; PairRanks: TPairRankMap): string; var
Chars: TStringArray;
Pairs: TStringArray;
Best: string;
BestRank: integer;
i, idx: integer;
NewList: TStringList;
parts: TStringArray;
a, b: string;
begin
Chars := Token.Split(''); if Length(Chars) = 0 then Exit(Token);

while True do
begin
Pairs := GetPairs(Chars); if Length(Pairs) = 0 then Break;



BestRank := MaxInt;
Best := '';
for i := 0 to High(Pairs) do
begin
  idx := PairRanks.IndexOf(Pairs[i]);
  if idx <> -1 then
    if PairRanks.Data[idx] < BestRank then
    begin
      BestRank := PairRanks.Data[idx];
      Best := Pairs[i];
    end;
end;
if Best = '' then Break;
NewList := TStringList.Create;
i := 0;
while i < Length(Chars) do
begin
  if (i < Length(Chars)-1) and ((Chars[i] + ' ' + Chars[i+1]) = Best) then
  begin
    parts := Best.Split(' ');
    NewList.Add(parts[0] + parts[1]);
    Inc(i, 2);
  end
  else
  begin
    NewList.Add(Chars[i]);
    Inc(i);
  end;
end;
SetLength(Chars, NewList.Count);
for i := 0 to NewList.Count - 1 do
  Chars[i] := NewList[i];
NewList.Free;
end;

Result := String.Join(' ', Chars); end;

function Encode(const FileName: string; Vocab: TStringIntMap; PairRanks: TPairRankMap): TIVector;
var p, Text: string; Bytes: TBytes; Tokens: array of string; i: integer; BPETok: string; Parts: TStringArray; idx: integer; begin SetLength(Result, 0);

Text := ReadFileToString(FileName); Bytes := BytesOf(Text);

SetLength(Tokens, Length(Bytes)); for i := 0 to High(Bytes) do Tokens[i] := Chr(Bytes[i]);

for i := 0 to High(Tokens) do begin BPETok := BPE(Tokens[i], PairRanks); Parts := BPETok.Split(' ');



for p in Parts do
begin
  idx := Vocab.IndexOf(p);
  if idx = -1 then
    Writeln('Warning: unknown token "', p, '"');
  SetLength(Result, Length(Result) + 1);
  Result[High(Result)] := Vocab.Data[idx];
end;
end; end;

{function Encode(const FileName: string; Vocab: TStringIntMap; PairRanks: TPairRankMap): TIVector;
var p: string;
var Text: string; Words: TStringArray; i: integer; BPETok: string; Parts: TStringArray; idx: integer; begin SetLength(Result, 0);

Text := ReadFileToString(FileName); Words := Text.Split(' ');

for i := 0 to High(Words) do begin BPETok := BPE(Words[i], PairRanks); Parts := BPETok.Split(' ');


for p in Parts do
begin
  idx := Vocab.IndexOf(p);
  if idx = -1 then
    Writeln('Warning: unknown token "', p, '"');

  SetLength(Result, Length(Result) + 1);
  Result[High(Result)] := Vocab.Data[idx];
end;}

procedure runbpe;
var
Vocab: TStringIntMap;
Merges: TPairList;
PairRanks: TPairRankMap;
IDs: array of integer;
x: Integer;

begin
  Vocab := LoadVocab('eleuvocab.json');    writeln('loadvocab done');readln;
  Merges := LoadMerges('eleumerges.txt');   writeln('merges done');readln;
  PairRanks := BuildPairRanks(Merges);     writeln('build done, start bela');

  IDs := Encode('bela.txt', Vocab, PairRanks);

  for x in IDs do Write(x, ' ');
  end;

end.


