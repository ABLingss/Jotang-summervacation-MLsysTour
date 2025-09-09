	.file	"tmpay8qkp9e.c"
# GNU C17 (Ubuntu 13.3.0-6ubuntu2~24.04) version 13.3.0 (x86_64-linux-gnu)
#	compiled by GNU C version 13.3.0, GMP version 6.3.0, MPFR version 4.2.1, MPC version 1.3.1, isl version isl-0.26-GMP

# GGC heuristics: --param ggc-min-expand=100 --param ggc-min-heapsize=131072
# options passed: -mtune=generic -march=x86-64 -O1 -fasynchronous-unwind-tables -fstack-protector-strong -fstack-clash-protection -fcf-protection
	.text
	.globl	matrix_multiply_asm
	.type	matrix_multiply_asm, @function
matrix_multiply_asm:
.LFB0:
	.cfi_startproc
	endbr64	
# /tmp/tmpay8qkp9e.c:4:     for (int i = 0; i < n; i++) {
	testl	%ecx, %ecx	# n
	jle	.L9	#,
# /tmp/tmpay8qkp9e.c:3: double matrix_multiply_asm(double* A, double* B, double* C, int n) {
	pushq	%r14	#
	.cfi_def_cfa_offset 16
	.cfi_offset 14, -16
	pushq	%r13	#
	.cfi_def_cfa_offset 24
	.cfi_offset 13, -24
	pushq	%r12	#
	.cfi_def_cfa_offset 32
	.cfi_offset 12, -32
	pushq	%rbp	#
	.cfi_def_cfa_offset 40
	.cfi_offset 6, -40
	pushq	%rbx	#
	.cfi_def_cfa_offset 48
	.cfi_offset 3, -48
	movq	%rsi, %r12	# tmp117, B
	movq	%rdx, %rbp	# tmp118, C
	movl	%ecx, %ebx	# tmp119, n
	movslq	%ecx, %rsi	# n, n
	salq	$3, %rsi	#, _77
	movq	%rdi, %r11	# A, ivtmp.22
	addq	%rsi, %rdi	# _77, ivtmp.24
# /tmp/tmpay8qkp9e.c:4:     for (int i = 0; i < n; i++) {
	movl	$0, %r13d	#, i
# /tmp/tmpay8qkp9e.c:5:         for (int j = 0; j < n; j++) {
	movl	$0, %r14d	#, j
	jmp	.L3	#
.L6:
	movl	%eax, %r10d	# j, j
.L5:
	movq	%r8, %rcx	# ivtmp.15, _71
# /tmp/tmpay8qkp9e.c:6:             C[i * n + j] = 0.0;
	movq	$0x000000000, (%r8)	#, MEM[(double *)_71]
	movq	%r9, %rdx	# ivtmp.17, ivtmp.9
	movq	%r11, %rax	# ivtmp.22, ivtmp.8
.L4:
# /tmp/tmpay8qkp9e.c:8:                 C[i * n + j] += A[i * n + k] * B[k * n + j];
	movsd	(%rax), %xmm0	# MEM[(double *)_54], MEM[(double *)_54]
	mulsd	(%rdx), %xmm0	# MEM[(double *)_55], tmp110
# /tmp/tmpay8qkp9e.c:8:                 C[i * n + j] += A[i * n + k] * B[k * n + j];
	addsd	(%rcx), %xmm0	# MEM[(double *)_71], tmp112
	movsd	%xmm0, (%rcx)	# tmp112, MEM[(double *)_71]
# /tmp/tmpay8qkp9e.c:7:             for (int k = 0; k < n; k++) {
	addq	$8, %rax	#, ivtmp.8
	addq	%rsi, %rdx	# _77, ivtmp.9
	cmpq	%rdi, %rax	# ivtmp.24, ivtmp.8
	jne	.L4	#,
# /tmp/tmpay8qkp9e.c:5:         for (int j = 0; j < n; j++) {
	leal	1(%r10), %eax	#, j
# /tmp/tmpay8qkp9e.c:5:         for (int j = 0; j < n; j++) {
	addq	$8, %r8	#, ivtmp.15
	addq	$8, %r9	#, ivtmp.17
	cmpl	%eax, %ebx	# j, n
	jne	.L6	#,
# /tmp/tmpay8qkp9e.c:4:     for (int i = 0; i < n; i++) {
	leal	1(%r13), %eax	#, i
# /tmp/tmpay8qkp9e.c:4:     for (int i = 0; i < n; i++) {
	addq	%rsi, %r11	# _77, ivtmp.22
	addq	%rsi, %rdi	# _77, ivtmp.24
	addq	%rsi, %rbp	# _77, ivtmp.25
	cmpl	%r10d, %r13d	# j, i
	je	.L2	#,
	movl	%eax, %r13d	# i, i
.L3:
	movq	%r12, %r9	# B, ivtmp.17
	movq	%rbp, %r8	# ivtmp.25, ivtmp.15
# /tmp/tmpay8qkp9e.c:5:         for (int j = 0; j < n; j++) {
	movl	%r14d, %r10d	# j, j
	jmp	.L5	#
.L2:
# /tmp/tmpay8qkp9e.c:13: }
	pxor	%xmm0, %xmm0	#
	popq	%rbx	#
	.cfi_def_cfa_offset 40
	popq	%rbp	#
	.cfi_def_cfa_offset 32
	popq	%r12	#
	.cfi_def_cfa_offset 24
	popq	%r13	#
	.cfi_def_cfa_offset 16
	popq	%r14	#
	.cfi_def_cfa_offset 8
	ret	
.L9:
	.cfi_restore 3
	.cfi_restore 6
	.cfi_restore 12
	.cfi_restore 13
	.cfi_restore 14
	pxor	%xmm0, %xmm0	#
	ret	
	.cfi_endproc
.LFE0:
	.size	matrix_multiply_asm, .-matrix_multiply_asm
	.ident	"GCC: (Ubuntu 13.3.0-6ubuntu2~24.04) 13.3.0"
	.section	.note.GNU-stack,"",@progbits
	.section	.note.gnu.property,"a"
	.align 8
	.long	1f - 0f
	.long	4f - 1f
	.long	5
0:
	.string	"GNU"
1:
	.align 8
	.long	0xc0000002
	.long	3f - 2f
2:
	.long	0x3
3:
	.align 8
4:
