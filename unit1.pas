unit Unit1;

{$mode ObjFPC}{$H+}

interface

implementation

type
  TTestVector = array[0..99] of Single;
var
  TestVector: TTestVector;

procedure InitTestVector(var N: TTestVector);
var
  i: Integer;
begin
for i := 0 to 99 do
  N[i] := 0.0;
end;


end.

