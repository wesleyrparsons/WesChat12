unit Matrix;

{$mode ObjFPC}{$H+}{$I proprietary.txt}

{ WesChat, Version 1.1, January 10, 2026, by Wesley R. Parsons, wespar@bellouth.net, www.wespar.com.}

interface

uses
  Global,
  Math;

type
  // MKL_INT is 32-bit int in CBLAS interface.
  TMKLInt = LongInt;

const
  CBlasRowMajor = 101;
  CBlasNoTrans = 111;
  CBlasTrans = 112;

procedure MatMulFull(const A, B: PSingle; C: PSingle; M, N, K, lda, ldb, ldc: Integer);
procedure MatMulFullTX(const A, B: PSingle; C: PSingle; M, N, K, lda, ldb, ldc: Integer);
procedure MatMulFullXT(const A, B: PSingle; C: PSingle; M, N, K, lda, ldb, ldc: Integer);
procedure MatMul(const A, B: PSingle; C: PSingle; M, N, K: Integer);
procedure MatMulXT(const A, B: PSingle; C: PSingle; M, N, K: Integer);
procedure MatMulTX(const A, B: PSingle; C: PSingle; M, N, K: Integer);
procedure AddMatVec(A, B: PSingle; M, N: Integer);
procedure MatAdd(const A, B: TSeqMatrix; var C: TSeqMatrix; Rows, Cols: Integer);
procedure VerticalPartitionX(const X: TSeqMatrix; const h: Integer; var XHead: TSeqHeadMatrix);
procedure VerticalConcatX(const XHead: array of TSeqHeadMatrix; const h: Integer; var X: TSeqMatrix);
procedure GradSplit(const Upstream: TSeqMatrix; var Left, Right: TSeqMatrix; Rows, Cols: Integer);
procedure AddGrad(const A, B: TSeqMatrix; var C: TSeqMatrix; Rows, Cols: Integer);
procedure AccumulateGrad(const src: TSeqMatrix; var dst: TSeqMatrix);
procedure MatMulAccXT(const A, B: PSingle; C: PSingle; M, N, K: Integer);
procedure ReLUMatrix(const A: THiddenMatrix; var B: THiddenMatrix);
procedure ReLUMatrixBackwards(const Hidden: THiddenMatrix; var dHidden: THiddenMatrix);
procedure CopyXMatrix(const A: array of TSeqVector; var B: array of TSeqVector);
function GELUMatrix(var A: THiddenMatrix): THiddenMatrix;

procedure cblas_sgemm(
    Layout: LongInt;
    TransA: LongInt;
    TransB: LongInt;
    M: TMKLInt;
    N: TMKLInt;
    K: TMKLInt;
    Alpha: Single;
    const A: PSingle;
    LDA: TMKLInt;
    const B: PSingle;
    LDB: TMKLInt;
    Beta: Single;
    C: PSingle;
    LDC: TMKLInt
); cdecl; external 'libopenblas.dll';

procedure cblas_saxpy(
    N: LongInt;
    alpha: Single;
    X: PSingle;
    incX: LongInt;
    Y: PSingle;
    incY: LongInt
); cdecl; external 'libopenblas.dll';

procedure cblas_scopy(
    N: LongInt;
    const X: PSingle;
    incX: LongInt;
    Y: PSingle;
    incY: LongInt
); cdecl; external 'libopenblas.dll';

procedure cblas_sscal(
    N: LongInt;
    alpha: Single;
    X: PSingle;
    incX: LongInt
); cdecl; external 'libopenblas.dll';

function cblas_sdot(
    N: LongInt;
    const X: PSingle;
    incX: LongInt;
    const Y: PSingle;
    incY: LongInt
): Single; cdecl; external 'libopenblas.dll';

function cblas_snrm2(
    N: LongInt;
    const X: PSingle;
    incX: LongInt
): Single; cdecl; external 'libopenblas.dll';

implementation

procedure VerticalPartitionX(const X: TSeqMatrix; const h: Integer; var XHead: TSeqHeadMatrix);
var
  i: Integer;
begin
//  for h := 0 to nHead - 1 do
    for i:= 0 to SeqLen - 1 do
      {for j:= 0 to HeadLen - 1 do
        XHead[h, i, j] := X[i, h * HeadLen + j];}
      cblas_scopy(HeadLen, @X[i, h * HeadLen], 1, @XHead[i, 0], 1);
end;

procedure VerticalConcatX(const XHead: array of TSeqHeadMatrix; const h: Integer; var X: TSeqMatrix);
var
  i: Integer;
begin
//  for h := 0 to nHead - 1 do
    for i := 0 to SeqLen - 1 do
      cblas_scopy(HeadLen, @XHead[i, 0], 1, @X[i, h * HeadLen], 1);
end;

// Split Gradient into 2 streams, for backprop.
procedure GradSplit(const Upstream: TSeqMatrix; var Left, Right: TSeqMatrix; Rows, Cols: Integer);
var
  N: Integer;
begin
  N := Rows * Cols;

  // Left += Upstream.
  cblas_saxpy(
    N,
    1.0,
    @Upstream[0,0], 1,
    @Left[0,0], 1);

  // Right += Upstream.
  cblas_saxpy(
    N,
    1.0,
    @Upstream[0,0], 1,
    @Right[0,0], 1);
end;

procedure AccumulateGrad(const src: TSeqMatrix; var dst: TSeqMatrix);
begin
  cblas_saxpy(XSize, 1.0, @src[0,0], 1, @dst[0,0], 1);
end;

procedure AddGrad(const A, B: TSeqMatrix; var C: TSeqMatrix; Rows, Cols: Integer);
var
  N: Integer;
begin
  N := Rows * Cols;

  // C := A.
  cblas_scopy(N,
              @A[0,0], 1,
              @C[0,0], 1);

  // C += B.
  AccumulateGrad(B, C);
end;

// Full matrix multiplication (lda, ldb, ldc), no transpose, overwrite, row-major.
procedure MatMulFull(const A, B: PSingle; C: PSingle; M, N, K, lda, ldb, ldc: Integer);
begin
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
    M, N, K,
    1.0,
    A, lda,
    B, ldb,
    0.0,
    C, ldc);
end;

// Full matrix multiplication (lda, ldb, ldc), A transpose, overwrite, row-major.
procedure MatMulFullXT(const A, B: PSingle; C: PSingle; M, N, K, lda, ldb, ldc: Integer);
begin
  cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
    M, N, K,
    1.0,
    A, lda,
    B, ldb,
    0.0,
    C, ldc);
end;

// Full matrix multiplication (lda, ldb, ldc), B transpose, overwrite, row-major.
procedure MatMulFullTX(const A, B: PSingle; C: PSingle; M, N, K, lda, ldb, ldc: Integer);
begin
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
    M, N, K,
    1.0,
    A, lda,
    B, ldb,
    0.0,
    C, ldc);
end;

// Matrix multiplication, no transpose, overwrite, row-major.
procedure MatMul(const A, B: PSingle; C: PSingle; M, N, K: Integer);
begin
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
       M, N, K,
       1.0,
       A, K,
       B, N,
       0.0,
       C, N);
end;

// Matrix multiplication, B transpose, overwrite, row-major.
procedure MatMulXT(const A, B: PSingle; C: PSingle; M, N, K: Integer);
begin
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
       M, N, K,
       1.0,
       A, K,
       B, K,
       0.0,
       C, N);
end;

// Matrix multiplication, A transpose, overwrite, row-major.
procedure MatMulTX(const A, B: PSingle; C: PSingle; M, N, K: Integer);
begin
  cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
       M, N, K,
       1.0,
       A, M,
       B, N,
       0.0,
       C, N);
end;

// Matrix multiplication, no transpose, accumulate, row-major.
procedure MatMulAcc(const A, B: PSingle; var C: PSingle; M, N, K: Integer);
begin
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
       M, N, K,
       1.0,
       A, K,
       B, N,
       1.0,
       C, N);
end;

// Matrix multiplication, B transpose, accumulate, row-major.
procedure MatMulAccXT(const A, B: PSingle; C: PSingle; M, N, K: Integer);
begin
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
       M, N, K,
       1.0,
       A, K,
       B, K,
       1.0,
       C, N);
end;

// Matrix addition, overwrite.
procedure MatAdd(const A, B: TSeqMatrix; var C: TSeqMatrix; Rows, Cols: Integer);
var
  N: Integer;
begin
  N := Rows * Cols;

  // C := A.
  cblas_scopy(
    N,
    @A[0,0], 1,
    @C[0,0], 1);

  // C += B.
  cblas_saxpy(
    N,
    1.0,
    @B[0,0], 1,
    @C[0,0], 1);
end;

// Matrix addition, accumulate, for adding B* to W*.
// Not used.
procedure AddMatVec(A, B: PSingle; M, N: Integer);
var
  i: Integer;
begin
  for i := 0 to M - 1 do begin
    cblas_saxpy(N, 1.0, B, 1, A, 1);
    Inc(A, N);
  end;
end;

// Matrix addition, accumulate.
procedure MatAccumulate(const A: TSeqMatrix; var C: TSeqMatrix; Rows, Cols: Integer);
var
  N: Integer;
begin
  N := Rows * Cols;

  // C += A.
  cblas_saxpy(
    N,
    1.0,
    @A[0,0], 1,
    @C[0,0], 1);
end;

{For each row, you need: Xstd = X − μσ. Where:
   μ = mean of the column.
   σ = standard deviation (usually computed with L2 norm).
 This is the standard z-score transform.}
procedure StandardizeRows(var A: array of Single; M, N: Integer);
var
  r, j: Integer;
  mean, std, invStd: Single;
  Ones: array of Single;
begin
  // Build a vector of ones for mean calculation.
  SetLength(Ones, N);
  for j := 0 to N - 1 do
    Ones[j] := 1.0;

  for r := 0 to M - 1 do begin
    // Pointer to the start of row r.
    // Row r begins at index r * N.
    // Row is contiguous, so stride = 1.
    // 1. Compute mean = (1/N) * sum(row).
    mean := cblas_sdot(N, @A[r * N], 1, @Ones[0], 1) / N;

    // 2. Subtract mean: row := row - mean.
    cblas_saxpy(N, -mean, @Ones[0], 1, @A[r * N], 1);

    // 3. Compute std = sqrt(sum((row - mean)^2) / N).
    std := cblas_snrm2(N, @A[r * N], 1) / Sqrt(N);

    // Avoid divide-by-zero.
    if std > 1e-12 then begin
      invStd := 1.0 / std;

    // 4. Scale row: row := row * (1 / std).
    cblas_sscal(N, invStd, @A[r * N], 1);
    end;
  end;
end;

// Apply ReLU to each iterm in a matrix.
procedure ReLUMatrix(const A: THiddenMatrix; var B: THiddenMatrix);
var
  i, j: Integer;
begin
  for i:= 0 to High(A) do
    for j := 0 to High(A[0]) do
      B[i, j] := Max(0.0, A[i, j]);
end;

procedure ReLUMatrixBackwards(const Hidden: THiddenMatrix; var dHidden: THiddenMatrix);
var
  i, j: Integer;
begin
  for i := 0 to High(Hidden) do
    for j := 0 to High(Hidden[0]) do
      if Hidden[i, j] <= 0.0 then
        dHidden[i, j] := 0.0;
end;

// GELU function.
function Gelu(x: Single): Single;
const
  SQRT_2_OVER_PI = 0.7978845608028654;   // Sqrt(2 / pi).
begin
  Result := 0.5 * x * (1.0 + TanH(SQRT_2_OVER_PI * (x + 0.044715 * x * x * x)));
end;

// Apply GELU to matrix. Chnage this to a procedure.
function GELUMatrix(var A: THiddenMatrix): THiddenMatrix;
var
  i, j: Integer;
begin
  for i:= 0 to SeqLen - 1 do
    for j := 0 to ModelDimProj - 1 do
      Result[i, j] := Gelu(A[i, j]);
end;

// Copy X matrix.
function OldCopyXMatrix(var A: array of TSeqVector): TSeqMatrix;
var
  i, j: Integer;
begin
  for i:= 0 to High(A) do
    for j := 0 to High(A[i]) do
      Result[i, j] := A[i, j];
end;

procedure CopyXMatrix(const A: array of TSeqVector; var B: array of TSeqVector);
var
  i: Integer;
begin
  for i := 0 to SeqLen - 1 do
    cblas_scopy(Length(A[0]), @A[i, 0], 1, @B[i, 0], 1);
end;

procedure SplitResidual(const upstream: TSeqMatrix; var left, right: TSeqMatrix);
var
  i, j: Integer;
  rows, cols: Integer;
begin
  rows := Length(upstream);
  if rows = 0 then Exit;

  cols := Length(upstream[0]);

  for i := 0 to rows - 1 do
    for j := 0 to cols - 1 do begin
      left[i, j]  := left[i, j]  + upstream[i, j];
      right[i, j] := right[i, j] + upstream[i, j];
    end;
end;

procedure AddToMatrix(var dst: TSeqMatrix; const src: TSeqMatrix);
var
  i, j: Integer;
  rows, cols: Integer;
begin
  rows := Length(dst);
  if rows = 0 then Exit;

  cols := Length(dst[0]);

  for i := 0 to rows - 1 do
    for j := 0 to cols - 1 do
      dst[i, j] := dst[i, j] + src[i, j];
end;

end.

