	.file	"tmphjiufdgd.c"
# GNU C17 (Ubuntu 13.3.0-6ubuntu2~24.04) version 13.3.0 (x86_64-linux-gnu)
#	compiled by GNU C version 13.3.0, GMP version 6.3.0, MPFR version 4.2.1, MPC version 1.3.1, isl version isl-0.26-GMP

# GGC heuristics: --param ggc-min-expand=100 --param ggc-min-heapsize=131072
# options passed: -mtune=generic -march=x86-64 -O2 -fasynchronous-unwind-tables -fstack-protector-strong -fstack-clash-protection -fcf-protection
	.text
	.p2align 4
	.globl	matrix_multiply_asm
	.type	matrix_multiply_asm, @function
matrix_multiply_asm:
.LFB0:
	.cfi_startproc
	endbr64	
# /tmp/tmphjiufdgd.c:4:     for (int i = 0; i < n; i++) {
	testl	%ecx, %ecx	# n
	jle	.L9	#,
# /tmp/tmphjiufdgd.c:3: double matrix_multiply_asm(double* A, double* B, double* C, int n) {
	pushq	%r14	#
	.cfi_def_cfa_offset 16
	.cfi_offset 14, -16
	movslq	%ecx, %r14	# n, _80
	movq	%rdi, %r10	# tmp115, A
	movl	%ecx, %r11d	# tmp118, n
	pushq	%r13	#
	.cfi_def_cfa_offset 24
	.cfi_offset 13, -24
	movq	%rdx, %r13	# tmp117, C
	pushq	%r12	#
	.cfi_def_cfa_offset 32
	.cfi_offset 12, -32
	movq	%rsi, %r12	# tmp116, B
	leaq	0(,%r14,8), %rsi	#, _81
	pushq	%rbp	#
	.cfi_def_cfa_offset 40
	.cfi_offset 6, -40
	leaq	(%rdi,%rsi), %rdi	#, ivtmp.25
# /tmp/tmphjiufdgd.c:4:     for (int i = 0; i < n; i++) {
	xorl	%ebp, %ebp	# i
# /tmp/tmphjiufdgd.c:3: double matrix_multiply_asm(double* A, double* B, double* C, int n) {
	pushq	%rbx	#
	.cfi_def_cfa_offset 48
	.cfi_offset 3, -48
# /tmp/tmphjiufdgd.c:4:     for (int i = 0; i < n; i++) {
	xorl	%ebx, %ebx	# ivtmp.27
	.p2align 4,,10
	.p2align 3
.L3:
	leaq	0(%r13,%rbx,8), %rcx	#, ivtmp.16
	movq	%r12, %r8	# B, ivtmp.18
# /tmp/tmphjiufdgd.c:5:         for (int j = 0; j < n; j++) {
	xorl	%r9d, %r9d	# j
	.p2align 4,,10
	.p2align 3
.L5:
# /tmp/tmphjiufdgd.c:6:             C[i * n + j] = 0.0;
	movq	$0x000000000, (%rcx)	#, MEM[(double *)_76]
	movq	%r8, %rdx	# ivtmp.18, ivtmp.10
	movq	%r10, %rax	# ivtmp.23, ivtmp.9
	pxor	%xmm1, %xmm1	# _19
	.p2align 4,,10
	.p2align 3
.L4:
# /tmp/tmphjiufdgd.c:8:                 C[i * n + j] += A[i * n + k] * B[k * n + j];
	movsd	(%rax), %xmm0	# MEM[(double *)_37], MEM[(double *)_37]
	mulsd	(%rdx), %xmm0	# MEM[(double *)_36], tmp110
# /tmp/tmphjiufdgd.c:7:             for (int k = 0; k < n; k++) {
	addq	$8, %rax	#, ivtmp.9
	addq	%rsi, %rdx	# _81, ivtmp.10
# /tmp/tmphjiufdgd.c:8:                 C[i * n + j] += A[i * n + k] * B[k * n + j];
	addsd	%xmm0, %xmm1	# tmp110, _19
	movsd	%xmm1, (%rcx)	# _19, MEM[(double *)_76]
# /tmp/tmphjiufdgd.c:7:             for (int k = 0; k < n; k++) {
	cmpq	%rdi, %rax	# ivtmp.25, ivtmp.9
	jne	.L4	#,
# /tmp/tmphjiufdgd.c:5:         for (int j = 0; j < n; j++) {
	leal	1(%r9), %eax	#, j
# /tmp/tmphjiufdgd.c:5:         for (int j = 0; j < n; j++) {
	addq	$8, %rcx	#, ivtmp.16
	addq	$8, %r8	#, ivtmp.18
	cmpl	%eax, %r11d	# j, n
	je	.L13	#,
	movl	%eax, %r9d	# j, j
	jmp	.L5	#
.L13:
# /tmp/tmphjiufdgd.c:4:     for (int i = 0; i < n; i++) {
	leal	1(%rbp), %eax	#, i
# /tmp/tmphjiufdgd.c:4:     for (int i = 0; i < n; i++) {
	addq	%rsi, %r10	# _81, ivtmp.23
	addq	%rsi, %rdi	# _81, ivtmp.25
	addq	%r14, %rbx	# _80, ivtmp.27
	cmpl	%r9d, %ebp	# j, i
	je	.L2	#,
	movl	%eax, %ebp	# i, i
	jmp	.L3	#
.L2:
# /tmp/tmphjiufdgd.c:13: }
	popq	%rbx	#
	.cfi_def_cfa_offset 40
	pxor	%xmm0, %xmm0	#
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
