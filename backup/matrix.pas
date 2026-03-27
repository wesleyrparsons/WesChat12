unit Matrix;

{$mode ObjFPC}{$H+}{$I proprietary.txt}

{ WesChat, Version 1.2, begun January 10, 2026, by Wesley R. Parsons, wespar@bellouth.net, www.wespar.com.}

interface

uses
  Global,
  Math;

type
  // MKL_INT is 32-bit int in CBLAS interface.
  TMKLInt = LongInt;

const
  CBlasRowMajor = 101;
  CBlasNoTrans  = 111;
  CBlasTrans    = 112;

// Multiply and add procedures.
procedure MatMulFullNN(const A, B: PSingle; C: PSingle; M, N, K, lda, ldb, ldc: Integer);
procedure MatMulFullTN(const A, B: PSingle; C: PSingle; M, N, K, lda, ldb, ldc: Integer);
procedure MatMulFullNT(const A, B: PSingle; C: PSingle; M, N, K, lda, ldb, ldc: Integer);
procedure MatMulNN(const A, B: PSingle; C: PSingle; M, N, K: Integer);
procedure MatMulNT(const A, B: PSingle; C: PSingle; M, N, K: Integer);
procedure MatMulTN(const A, B: PSingle; C: PSingle; M, N, K: Integer);
procedure MatAdd(const A, B: TSeqMatrix; var C: TSeqMatrix; Rows, Cols: Integer);

// Partition and concatenate procedures.
procedure VerticalPartitionX(const X: TSeqMatrix; const h: Integer; var XHead: TSeqHeadMatrix; const L, HL: Integer);
procedure VerticalConcatX(const XHead: array of TSeqHeadMatrix; const h: Integer; var X: TSeqMatrix; const L, HL: Integer);

// Split and accumulate procedures.
procedure GradSplit(const Upstream: TSeqMatrix; var Left, Right: TSeqMatrix; Rows, Cols: Integer);
procedure AccumulateGrad(const Src: TSeqMatrix; var Dst: TSeqMatrix; Rows, Cols: Integer);
procedure MatMulAccNT(const A, B: PSingle; C: PSingle; M, N, K: Integer);
procedure MatMulAccNN(const A, B: PSingle; C: PSingle; M, N, K: Integer);

// ReLU procedures.
procedure ReLUMaskForward(const A: THiddenMatrix; var B: THiddenMatrix);
procedure ReLUMaskBackward(const Hidden: THiddenMatrix; var dHidden: THiddenMatrix);

// Copy matrix procedure.
procedure CopyXMatrix(const A: array of TSeqVector; var B: array of TSeqVector;
  const Rows, Cols: Integer);

// cblas sgemm.
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

// cblas saxpy.
procedure cblas_saxpy(
    N: LongInt;
    alpha: Single;
    X: PSingle;
    incX: LongInt;
    Y: PSingle;
    incY: LongInt
); cdecl; external 'libopenblas.dll';

// cblas scopy.
procedure cblas_scopy(
    N: LongInt;
    const X: PSingle;
    incX: LongInt;
    Y: PSingle;
    incY: LongInt
); cdecl; external 'libopenblas.dll';

// cblas sscal.
procedure cblas_sscal(
    N: LongInt;
    alpha: Single;
    X: PSingle;
    incX: LongInt
); cdecl; external 'libopenblas.dll';

// cblas sdot.
function cblas_sdot(
    N: LongInt;
    const X: PSingle;
    incX: LongInt;
    const Y: PSingle;
    incY: LongInt
): Single; cdecl; external 'libopenblas.dll';

// cblas snrm2.
function cblas_snrm2(
    N: LongInt;
    const X: PSingle;
    incX: LongInt
): Single; cdecl; external 'libopenblas.dll';

implementation

// Parition X into h heads.
procedure VerticalPartitionX(const X: TSeqMatrix; const h: Integer; var XHead: TSeqHeadMatrix; const L, HL: Integer);
var
  i: Integer;
begin
    for i:= 0 to L - 1 do
      cblas_scopy(HL, @X[i, h * HL], 1, @XHead[i, 0], 1);
end;

// Concatenate the h heads back into X.
procedure VerticalConcatX(const XHead: array of TSeqHeadMatrix; const h: Integer; var X: TSeqMatrix; const L, HL: Integer);
var
  i: Integer;
begin
  for i := 0 to L - 1 do
    cblas_scopy(HL, @XHead[h][i, 0], 1, @X[i, h * HL], 1);
end;

// Split Gradient into 2 streams, for backprop.
procedure GradSplit(const Upstream: TSeqMatrix; var Left, Right: TSeqMatrix; Rows, Cols: Integer);
var
  n: Integer;
begin
  n := Rows * Cols;

  // Left += Upstream.
  cblas_saxpy(
    n,
    1.0,
    @Upstream[0,0], 1,
    @Left[0,0], 1);

  // Right += Upstream.
  cblas_saxpy(
    n,
    1.0,
    @Upstream[0,0], 1,
    @Right[0,0], 1);
end;

// Accummulate Gradient.
procedure AccumulateGrad(const Src: TSeqMatrix; var Dst: TSeqMatrix; Rows, Cols: Integer);
var
  n: Integer;
begin
  n := Rows * Cols;
  cblas_saxpy(
  n,
  1.0,
  @Src[0,0], 1,
  @Dst[0,0], 1);
end;

{procedure AccumulateGrad(const src: TSeqMatrix; var dst: TSeqMatrix);               // Don't use XSize.
begin
  cblas_saxpy(XSize, 1.0, @src[0,0], 1, @dst[0,0], 1);
end; }

// Full matrix multiplication (lda, ldb, ldc), no transpose, overwrite, row-major.
procedure MatMulFullNN(const A, B: PSingle; C: PSingle; M, N, K, lda, ldb, ldc: Integer);
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
procedure MatMulFullNT(const A, B: PSingle; C: PSingle; M, N, K, lda, ldb, ldc: Integer);
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
procedure MatMulFullTN(const A, B: PSingle; C: PSingle; M, N, K, lda, ldb, ldc: Integer);
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
procedure MatMulNN(const A, B: PSingle; C: PSingle; M, N, K: Integer);
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
procedure MatMulNT(const A, B: PSingle; C: PSingle; M, N, K: Integer);
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
procedure MatMulTN(const A, B: PSingle; C: PSingle; M, N, K: Integer);
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
procedure MatMulAccNN(const A, B: PSingle; C: PSingle; M, N, K: Integer);
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
procedure MatMulAccNT(const A, B: PSingle; C: PSingle; M, N, K: Integer);
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
  n: Integer;
begin
  n := Rows * Cols;

  // C := A.
  cblas_scopy(
    n,
    @A[0,0], 1,
    @C[0,0], 1);

  // C += B.
  cblas_saxpy(
    n,
    1.0,
    @B[0,0], 1,
    @C[0,0], 1);
end;

// Matrix addition, accumulate.
procedure MatAccumulate(const A: TSeqMatrix; var C: TSeqMatrix; Rows, Cols: Integer);
var
  n: Integer;
begin
  n := Rows * Cols;

  // C += A.
  cblas_saxpy(
    n,
    1.0,
    @A[0,0], 1,
    @C[0,0], 1);
end;

{ Standard z-score transform: Xstd = X − μσ.
   μ = mean of the row.
   σ = standard deviation (usually computed with L2 norm).}
procedure StandardizeRows(var A: array of Single; M, N: Integer);
var
  r, j: Integer;
  Mean, Std, InvStd: Single;
  Ones: TFVector;
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
    Mean := cblas_sdot(N, @A[r * N], 1, @Ones[0], 1) / N;

    // 2. Subtract mean: row := row - mean.
    cblas_saxpy(N, -Mean, @Ones[0], 1, @A[r * N], 1);

    // 3. Compute std = sqrt(sum((row - mean)^2) / N).
    Std := cblas_snrm2(N, @A[r * N], 1) / Sqrt(N);

    // Avoid divide-by-zero.
    if Std > 1e-12 then begin
      InvStd := 1.0 / Std;

    // 4. Scale row: row := row * (1 / std).
    cblas_sscal(N, InvStd, @A[r * N], 1);
    end;
  end;
end;

// Apply ReLU to each iterm in a matrix.
procedure ReLUMaskForward(const A: THiddenMatrix; var B: THiddenMatrix);
var
  i, j: Integer;
begin
  for i:= 0 to High(A) do
    for j := 0 to High(A[0]) do
      B[i, j] := Max(0.0, A[i, j]);
end;

procedure ReLUMaskBackward(const Hidden: THiddenMatrix; var dHidden: THiddenMatrix);
var
  i, j: Integer;
begin
  for i := 0 to High(Hidden) do
    for j := 0 to High(Hidden[0]) do
      if Hidden[i, j] <= 0.0 then
        dHidden[i, j] := 0.0;
end;

procedure CopyXMatrix(const A: array of TSeqVector; var B: array of TSeqVector;
  const Rows, Cols: Integer);
var
  i: Integer;
begin
  if Rows <= 0 then Exit;
  if Cols <= 0 then Exit;

  for i := 0 to Rows - 1 do
    cblas_scopy(Cols, @A[i, 0], 1, @B[i, 0], 1);
end;

end.

