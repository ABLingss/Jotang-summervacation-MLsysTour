	.text
	.file	"tmpwnywvjth.c"
	.globl	matrix_multiply_asm             # -- Begin function matrix_multiply_asm
	.p2align	4, 0x90
	.type	matrix_multiply_asm,@function
matrix_multiply_asm:                    # @matrix_multiply_asm
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	%rdx, -24(%rbp)
	movl	%ecx, -28(%rbp)
	movl	$0, -32(%rbp)
.LBB0_1:                                # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_3 Depth 2
                                        #       Child Loop BB0_5 Depth 3
	movl	-32(%rbp), %eax
	cmpl	-28(%rbp), %eax
	jge	.LBB0_12
# %bb.2:                                #   in Loop: Header=BB0_1 Depth=1
	movl	$0, -36(%rbp)
.LBB0_3:                                #   Parent Loop BB0_1 Depth=1
                                        # =>  This Loop Header: Depth=2
                                        #       Child Loop BB0_5 Depth 3
	movl	-36(%rbp), %eax
	cmpl	-28(%rbp), %eax
	jge	.LBB0_10
# %bb.4:                                #   in Loop: Header=BB0_3 Depth=2
	movq	-24(%rbp), %rax
	movl	-32(%rbp), %ecx
	imull	-28(%rbp), %ecx
	addl	-36(%rbp), %ecx
	movslq	%ecx, %rcx
	xorps	%xmm0, %xmm0
	movsd	%xmm0, (%rax,%rcx,8)
	movl	$0, -40(%rbp)
.LBB0_5:                                #   Parent Loop BB0_1 Depth=1
                                        #     Parent Loop BB0_3 Depth=2
                                        # =>    This Inner Loop Header: Depth=3
	movl	-40(%rbp), %eax
	cmpl	-28(%rbp), %eax
	jge	.LBB0_8
# %bb.6:                                #   in Loop: Header=BB0_5 Depth=3
	movq	-8(%rbp), %rax
	movl	-32(%rbp), %ecx
	imull	-28(%rbp), %ecx
	addl	-40(%rbp), %ecx
	movslq	%ecx, %rcx
	movsd	(%rax,%rcx,8), %xmm0            # xmm0 = mem[0],zero
	movq	-16(%rbp), %rax
	movl	-40(%rbp), %ecx
	imull	-28(%rbp), %ecx
	addl	-36(%rbp), %ecx
	movslq	%ecx, %rcx
	movsd	(%rax,%rcx,8), %xmm2            # xmm2 = mem[0],zero
	movq	-24(%rbp), %rax
	movl	-32(%rbp), %ecx
	imull	-28(%rbp), %ecx
	addl	-36(%rbp), %ecx
	movslq	%ecx, %rcx
	movsd	(%rax,%rcx,8), %xmm1            # xmm1 = mem[0],zero
	mulsd	%xmm2, %xmm0
	addsd	%xmm1, %xmm0
	movsd	%xmm0, (%rax,%rcx,8)
# %bb.7:                                #   in Loop: Header=BB0_5 Depth=3
	movl	-40(%rbp), %eax
	addl	$1, %eax
	movl	%eax, -40(%rbp)
	jmp	.LBB0_5
.LBB0_8:                                #   in Loop: Header=BB0_3 Depth=2
	jmp	.LBB0_9
.LBB0_9:                                #   in Loop: Header=BB0_3 Depth=2
	movl	-36(%rbp), %eax
	addl	$1, %eax
	movl	%eax, -36(%rbp)
	jmp	.LBB0_3
.LBB0_10:                               #   in Loop: Header=BB0_1 Depth=1
	jmp	.LBB0_11
.LBB0_11:                               #   in Loop: Header=BB0_1 Depth=1
	movl	-32(%rbp), %eax
	addl	$1, %eax
	movl	%eax, -32(%rbp)
	jmp	.LBB0_1
.LBB0_12:
	xorps	%xmm0, %xmm0
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end0:
	.size	matrix_multiply_asm, .Lfunc_end0-matrix_multiply_asm
	.cfi_endproc
                                        # -- End function
	.ident	"Ubuntu clang version 18.1.3 (1ubuntu1)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
