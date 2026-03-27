unit Extras;

{$mode ObjFPC}{$H+}{$I proprietary.txt}

{ WesChat, Version 1.2, begun January 10, 2026, by Wesley R. Parsons, wespar@bellouth.net, www.wespar.com.}

interface

implementation

// Welford addition algorithm. Not used yet.
procedure WelfordAddition(const X: TSeqMatrix; SeqLen, ModelDim: Integer; out MeanX, VarX: TFVector);
var
  t, d: Integer;
  Count: Int64;
  Entry, Delta, Delta2: Single;
begin // X[t, d]
  // Initialize.
  for d := 0 to ModelDim - 1 do begin
    MeanX[d] := 0.0;
    VarX[d] := 0.0;   // M2 accumulator.
  end;

  Count := 0;

  // Sweep through sequence.
  for t := 0 to SeqLen - 1 do begin
    Inc(Count);

    for d := 0 to ModelDim - 1 do begin
      Entry := X[t, d];
      Delta  := Entry - MeanX[d];
      MeanX[d] := MeanX[d] + Delta / Count;
      Delta2 := Entry - MeanX[d];
      VarX[d] := VarX[d] + Delta * Delta2;
    end;
  end;

  // Finalize variance.
  if Count > 1 then
    for d := 0 to ModelDim - 1 do
      VarX[d] := VarX[d] / (Count - 1)
  else
    for d := 0 to ModelDim - 1 do
      VarX[d] := 0.0;
end;

// Add B1 Vector to X-type matrix.
procedure AddB1Matrix(const A: THiddenMatrix; const B: TSeqVectorProj; var C: THiddenMatrix);
var
  i, j: Integer;
begin
  for i:= 0 to ModelDim - 1 do
    for j := 0 to ModelDimProj - 1 do
      C[i, j] := A[i, j] + B[j];
end;

// Add B2 Vector to X-type matrix.
procedure AddB2Matrix(const A: TSeqMatrix; const B: TSeqVector; var C: TSeqMatrix);
var
  i, j: Integer;
begin
  for i:= 0 to SeqLen - 1 do
    for j := 0 to ModelDim - 1 do
      C[i, j] := A[i, j] + B[i];
end;

// Adds A and B matrices to get C matrix, all L x D.
procedure AddLDMatrices(const A, B: TSeqMatrix; L: Integer; var C: TSeqMatrix);
var
  i, j: Integer;
begin
  for i:= 0 to L - 1 do
    for j := 0 to ModelDim - 1 do
      C[i, j] := A[i, j] + B[i, j];
end;

// Q * Wq is L x D, D x D, output K L x D
procedure MatMul(const A: SASHLDArray; B: SASHDDArray; out C: SASHLDArray);
var
  M, K, N: Integer;
  i, j, l: Integer;
  Sum: Single;
begin
  M := Length(A);
  K := Length(A[0]);
  N := Length(B[0]);

  for i := 0 to M - 1 do begin
    for j := 0 to N - 1 do begin
      sum := 0;
      for l := 0 to K - 1 do
        sum := sum + A[i, l] * B[l, j];
      C[i, j] := Sum;
    end;
  end;
end;

// F87 optimized blocked matrix operations.
procedure BlockedMatMulLLLD(const A: SASHLLArray; B: SASHLDArray; out C: SASHLDArray);
const
  BS = 32;  // block size
var
  M, K, N: Integer;
  i, j, l: Integer;
  ii, jj, kk: Integer;
  sum: Single;
begin
  M := Length(A);
  K := Length(A[0]);
  N := Length(B[0]);

  //SetLength(C, M, N);

  // initialize C
  for i := 0 to M - 1 do
    for j := 0 to N - 1 do
      C[i][j] := 0;

  ii := 0;
  while ii < M do begin
    kk := 0;
    while kk < K do begin
      jj := 0;
      while jj < N do begin
        for i := ii to Min(ii + BS - 1, M - 1) do
          for l := kk to Min(kk + BS - 1, K - 1) do begin
            sum := A[i][l];
            for j := jj to Min(jj + BS - 1, N - 1) do
              C[i][j] := C[i][j] + sum * B[l][j];
          end;

        jj := jj + BS;
      end;
      kk := kk + BS;
    end;
    ii := ii + BS;
  end;
end;

// F87 optimized blocked matrix operations.
procedure BlockedMatMulLDDD(const A: SASHLDArray; B: SASHDDArray; out C: SASHLDArray);
const
  BS = 32;  // block size
var
  M, K, N: Integer;
  i, j, l: Integer;
  ii, jj, kk: Integer;
  sum: Single;
begin
  M := Length(A);
  K := Length(A[0]);
  N := Length(B[0]);

  //SetLength(C, M, N);

  // initialize C
  for i := 0 to M - 1 do
    for j := 0 to N - 1 do
      C[i][j] := 0;

  ii := 0;
  while ii < M do begin
    kk := 0;
    while kk < K do begin
      jj := 0;
      while jj < N do begin
        for i := ii to Min(ii + BS - 1, M - 1) do
          for l := kk to Min(kk + BS - 1, K - 1) do begin
            sum := A[i][l];
            for j := jj to Min(jj + BS - 1, N - 1) do
              C[i][j] := C[i][j] + sum * B[l][j];
          end;

        jj := jj + BS;
      end;
      kk := kk + BS;
    end;
    ii := ii + BS;
  end;
end;

// F87 optimized blocked matrix operations.
procedure BlockedMatMulLDDL(const A: SASHLDArray; B: SASHDLArray; out C: SASHLLArray);
const
  BS = 32;  // block size
var
  M, K, N: Integer;
  i, j, l: Integer;
  ii, jj, kk: Integer;
  sum: Single;
begin
  M := Length(A);
  K := Length(A[0]);
  N := Length(B[0]);

  //SetLength(C, M, N);

  // initialize C
  for i := 0 to M - 1 do
    for j := 0 to N - 1 do
      C[i][j] := 0;

  ii := 0;
  while ii < M do begin
    kk := 0;
    while kk < K do begin
      jj := 0;
      while jj < N do begin
        for i := ii to Min(ii + BS - 1, M - 1) do
          for l := kk to Min(kk + BS - 1, K - 1) do begin
            sum := A[i][l];
            for j := jj to Min(jj + BS - 1, N - 1) do
              C[i][j] := C[i][j] + sum * B[l][j];
          end;

        jj := jj + BS;
      end;
      kk := kk + BS;
    end;
    ii := ii + BS;
  end;
end;

// Trnspose matrix L and D dimenstions.
function TransposeLD(const A: SASHLDArray): SASHDLArray;
var
  M, N: Integer;
  i, j: Integer;
begin
  M := Length(A);
  N := Length(A[0]);

  //SetLength(AT, N, M);

  for i := 0 to M - 1 do
    for j := 0 to N - 1 do
      Result[j, i] := A[i, j];
end;

end.

