	.text
	.file	"tmpaju4rak8.c"
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0                          # -- Begin function matrix_multiply_asm
.LCPI0_0:
	.quad	0                               # 0x0
	.quad	1                               # 0x1
	.quad	2                               # 0x2
	.quad	3                               # 0x3
	.quad	4                               # 0x4
	.quad	5                               # 0x5
	.quad	6                               # 0x6
	.quad	7                               # 0x7
	.section	.rodata.cst8,"aM",@progbits,8
	.p2align	3, 0x0
.LCPI0_1:
	.quad	8                               # 0x8
.LCPI0_2:
	.quad	16                              # 0x10
.LCPI0_3:
	.quad	24                              # 0x18
.LCPI0_4:
	.quad	32                              # 0x20
	.text
	.globl	matrix_multiply_asm
	.p2align	4, 0x90
	.type	matrix_multiply_asm,@function
matrix_multiply_asm:                    # @matrix_multiply_asm
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	pushq	%r15
	.cfi_def_cfa_offset 24
	pushq	%r14
	.cfi_def_cfa_offset 32
	pushq	%r13
	.cfi_def_cfa_offset 40
	pushq	%r12
	.cfi_def_cfa_offset 48
	pushq	%rbx
	.cfi_def_cfa_offset 56
	subq	$40, %rsp
	.cfi_def_cfa_offset 96
	.cfi_offset %rbx, -56
	.cfi_offset %r12, -48
	.cfi_offset %r13, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	.cfi_offset %rbp, -16
	movq	%rdx, -48(%rsp)                 # 8-byte Spill
	movq	%rsi, -120(%rsp)                # 8-byte Spill
	movq	%rdi, -56(%rsp)                 # 8-byte Spill
	testl	%ecx, %ecx
	jle	.LBB0_51
# %bb.1:
	movq	-56(%rsp), %rdx                 # 8-byte Reload
	movl	%ecx, %eax
	movl	%ecx, %r15d
	vmovdqa64	.LCPI0_0(%rip), %zmm1   # zmm1 = [0,1,2,3,4,5,6,7]
	vpbroadcastq	.LCPI0_1(%rip), %zmm2   # zmm2 = [8,8,8,8,8,8,8,8]
	vpbroadcastq	.LCPI0_2(%rip), %zmm3   # zmm3 = [16,16,16,16,16,16,16,16]
	vpbroadcastq	.LCPI0_3(%rip), %zmm4   # zmm4 = [24,24,24,24,24,24,24,24]
	vpbroadcastq	.LCPI0_4(%rip), %zmm5   # zmm5 = [32,32,32,32,32,32,32,32]
	movl	$6661, %r9d                     # imm = 0x1A05
	leaq	(,%rax,8), %r11
	movq	%rax, %rbx
	vpbroadcastq	%rax, %zmm0
	shlq	$6, %rbx
	xorl	%esi, %esi
	movl	%r15d, -124(%rsp)               # 4-byte Spill
	bextrl	%r9d, %r15d, %ebp
	shlq	$8, %rbp
	leaq	(%rdx,%rax,8), %rcx
	leaq	192(%rdx), %rdi
	leaq	56(%rdx), %r8
	xorl	%edx, %edx
	movq	%rdi, -24(%rsp)                 # 8-byte Spill
	movq	-120(%rsp), %rdi                # 8-byte Reload
	movq	%r8, -32(%rsp)                  # 8-byte Spill
	movq	%rcx, -8(%rsp)                  # 8-byte Spill
	movabsq	$2305843009213693951, %rcx      # imm = 0x1FFFFFFFFFFFFFFF
	addq	%rax, %rcx
	imulq	%rax, %rcx
	movq	%rcx, -88(%rsp)                 # 8-byte Spill
	leaq	-1(%rax), %rcx
	movq	%rcx, -104(%rsp)                # 8-byte Spill
	movq	-48(%rsp), %rcx                 # 8-byte Reload
	leaq	(%rcx,%rax,8), %rcx
	movq	%rcx, -16(%rsp)                 # 8-byte Spill
	movl	%eax, %ecx
	andl	$2147483646, %ecx               # imm = 0x7FFFFFFE
	movq	%rcx, 32(%rsp)                  # 8-byte Spill
	movl	%eax, %ecx
	andl	$2147483616, %ecx               # imm = 0x7FFFFFE0
	movq	%rcx, -96(%rsp)                 # 8-byte Spill
	movl	%eax, %ecx
	andl	$7, %ecx
	movq	%rcx, -64(%rsp)                 # 8-byte Spill
	leaq	8(%rdi), %rcx
	movq	%rcx, -40(%rsp)                 # 8-byte Spill
	jmp	.LBB0_3
	.p2align	4, 0x90
.LBB0_2:                                #   in Loop: Header=BB0_3 Depth=1
	movq	24(%rsp), %rsi                  # 8-byte Reload
	movq	16(%rsp), %rdx                  # 8-byte Reload
	incq	%rdx
	addl	%r15d, %esi
	cmpq	%rax, %rdx
	je	.LBB0_51
.LBB0_3:                                # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_6 Depth 2
                                        #       Child Loop BB0_11 Depth 3
                                        #       Child Loop BB0_15 Depth 3
                                        #       Child Loop BB0_19 Depth 3
                                        #       Child Loop BB0_25 Depth 3
                                        #       Child Loop BB0_29 Depth 3
                                        #       Child Loop BB0_33 Depth 3
                                        #     Child Loop BB0_45 Depth 2
                                        #     Child Loop BB0_40 Depth 2
                                        #     Child Loop BB0_49 Depth 2
	movq	-32(%rsp), %r8                  # 8-byte Reload
	movq	%rsi, 24(%rsp)                  # 8-byte Spill
	movl	%esi, %esi
	movq	-24(%rsp), %r10                 # 8-byte Reload
	movq	-48(%rsp), %rdi                 # 8-byte Reload
	movq	%r11, %r14
	imulq	%rdx, %r14
	movl	%edx, %ecx
	movq	%rdx, 16(%rsp)                  # 8-byte Spill
	imulq	%rax, %rdx
	imull	%r15d, %ecx
	leaq	(%r8,%rsi,8), %r13
	movq	-56(%rsp), %r8                  # 8-byte Reload
	leaq	(%r10,%rsi,8), %r10
	leaq	(%rdi,%r14), %r12
	addq	-16(%rsp), %r14                 # 8-byte Folded Reload
	cmpq	$0, -104(%rsp)                  # 8-byte Folded Reload
	leaq	(%rdi,%rdx,8), %rdx
	movq	%r12, -80(%rsp)                 # 8-byte Spill
	leaq	(%r8,%rsi,8), %r9
	movq	-8(%rsp), %rsi                  # 8-byte Reload
	leaq	(%r8,%rcx,8), %r8
	movq	%r14, -72(%rsp)                 # 8-byte Spill
	movq	%r8, 8(%rsp)                    # 8-byte Spill
	leaq	(%rsi,%rcx,8), %rcx
	movq	%rcx, (%rsp)                    # 8-byte Spill
	je	.LBB0_34
# %bb.4:                                #   in Loop: Header=BB0_3 Depth=1
	cmpq	%rcx, %r12
	setb	%cl
	cmpq	%r14, %r8
	setb	%sil
	xorl	%r14d, %r14d
	andb	%cl, %sil
	movb	%sil, -125(%rsp)                # 1-byte Spill
	movq	-40(%rsp), %rsi                 # 8-byte Reload
	movq	%rsi, -112(%rsp)                # 8-byte Spill
	movq	-120(%rsp), %rsi                # 8-byte Reload
	jmp	.LBB0_6
	.p2align	4, 0x90
.LBB0_5:                                #   in Loop: Header=BB0_6 Depth=2
	addq	$16, -112(%rsp)                 # 8-byte Folded Spill
	movl	-124(%rsp), %r15d               # 4-byte Reload
	addq	$2, %r14
	addq	$16, %rsi
	cmpq	32(%rsp), %r14                  # 8-byte Folded Reload
	je	.LBB0_35
.LBB0_6:                                #   Parent Loop BB0_3 Depth=1
                                        # =>  This Loop Header: Depth=2
                                        #       Child Loop BB0_11 Depth 3
                                        #       Child Loop BB0_15 Depth 3
                                        #       Child Loop BB0_19 Depth 3
                                        #       Child Loop BB0_25 Depth 3
                                        #       Child Loop BB0_29 Depth 3
                                        #       Child Loop BB0_33 Depth 3
	vxorpd	%xmm6, %xmm6, %xmm6
	movq	$0, (%rdx,%r14,8)
	cmpl	$32, %r15d
	jb	.LBB0_7
# %bb.8:                                #   in Loop: Header=BB0_6 Depth=2
	movq	-88(%rsp), %rcx                 # 8-byte Reload
	movq	-120(%rsp), %r8                 # 8-byte Reload
	addq	%r14, %rcx
	leaq	(%r8,%r14,8), %rdi
	leaq	(%r8,%rcx,8), %rcx
	cmpq	%rcx, %rdi
	movq	%rcx, %r8
	cmovaq	%rdi, %r8
	cmovbq	%rdi, %rcx
	addq	$8, %r8
	cmpq	%r8, -80(%rsp)                  # 8-byte Folded Reload
	setb	%r8b
	cmpq	-72(%rsp), %rcx                 # 8-byte Folded Reload
	setb	%cl
	andb	%r8b, %cl
	orb	-125(%rsp), %cl                 # 1-byte Folded Reload
	je	.LBB0_10
.LBB0_7:                                #   in Loop: Header=BB0_6 Depth=2
	xorl	%r8d, %r8d
.LBB0_13:                               #   in Loop: Header=BB0_6 Depth=2
	movq	-64(%rsp), %r12                 # 8-byte Reload
	testq	%r12, %r12
	je	.LBB0_16
# %bb.14:                               #   in Loop: Header=BB0_6 Depth=2
	movq	%r11, %r15
	imulq	%r8, %r15
	movq	%r8, %rdi
	addq	%rsi, %r15
	.p2align	4, 0x90
.LBB0_15:                               #   Parent Loop BB0_3 Depth=1
                                        #     Parent Loop BB0_6 Depth=2
                                        # =>    This Inner Loop Header: Depth=3
	vmovsd	(%r15), %xmm7                   # xmm7 = mem[0],zero
	addq	%r11, %r15
	vfmadd231sd	(%r9,%rdi,8), %xmm7, %xmm6 # xmm6 = (xmm7 * mem) + xmm6
	incq	%rdi
	decq	%r12
	vmovsd	%xmm6, (%rdx,%r14,8)
	jne	.LBB0_15
	jmp	.LBB0_17
	.p2align	4, 0x90
.LBB0_16:                               #   in Loop: Header=BB0_6 Depth=2
	movq	%r8, %rdi
.LBB0_17:                               #   in Loop: Header=BB0_6 Depth=2
	movq	-104(%rsp), %rcx                # 8-byte Reload
	movl	-124(%rsp), %r15d               # 4-byte Reload
	subq	%r8, %rcx
	cmpq	$7, %rcx
	jb	.LBB0_20
# %bb.18:                               #   in Loop: Header=BB0_6 Depth=2
	movq	%r11, %r12
	imulq	%rdi, %r12
	addq	%rsi, %r12
	.p2align	4, 0x90
.LBB0_19:                               #   Parent Loop BB0_3 Depth=1
                                        #     Parent Loop BB0_6 Depth=2
                                        # =>    This Inner Loop Header: Depth=3
	vmovsd	(%r12), %xmm7                   # xmm7 = mem[0],zero
	leaq	(%r12,%rax,8), %r8
	vfmadd132sd	-56(%r13,%rdi,8), %xmm6, %xmm7 # xmm7 = (xmm7 * mem) + xmm6
	vmovsd	%xmm7, (%rdx,%r14,8)
	vmovsd	(%r12,%rax,8), %xmm6            # xmm6 = mem[0],zero
	addq	%rbx, %r12
	vfmadd132sd	-48(%r13,%rdi,8), %xmm7, %xmm6 # xmm6 = (xmm6 * mem) + xmm7
	vmovsd	%xmm6, (%rdx,%r14,8)
	vmovsd	(%r11,%r8), %xmm7               # xmm7 = mem[0],zero
	addq	%r11, %r8
	vfmadd132sd	-40(%r13,%rdi,8), %xmm6, %xmm7 # xmm7 = (xmm7 * mem) + xmm6
	vmovsd	%xmm7, (%rdx,%r14,8)
	vmovsd	(%r11,%r8), %xmm6               # xmm6 = mem[0],zero
	addq	%r11, %r8
	vfmadd132sd	-32(%r13,%rdi,8), %xmm7, %xmm6 # xmm6 = (xmm6 * mem) + xmm7
	vmovsd	%xmm6, (%rdx,%r14,8)
	vmovsd	(%r11,%r8), %xmm7               # xmm7 = mem[0],zero
	addq	%r11, %r8
	vfmadd132sd	-24(%r13,%rdi,8), %xmm6, %xmm7 # xmm7 = (xmm7 * mem) + xmm6
	vmovsd	%xmm7, (%rdx,%r14,8)
	vmovsd	(%r11,%r8), %xmm6               # xmm6 = mem[0],zero
	addq	%r11, %r8
	vfmadd132sd	-16(%r13,%rdi,8), %xmm7, %xmm6 # xmm6 = (xmm6 * mem) + xmm7
	vmovsd	%xmm6, (%rdx,%r14,8)
	vmovsd	(%r11,%r8), %xmm7               # xmm7 = mem[0],zero
	addq	%r11, %r8
	vfmadd132sd	-8(%r13,%rdi,8), %xmm6, %xmm7 # xmm7 = (xmm7 * mem) + xmm6
	vmovsd	%xmm7, (%rdx,%r14,8)
	vmovsd	(%r11,%r8), %xmm6               # xmm6 = mem[0],zero
	vfmadd132sd	(%r13,%rdi,8), %xmm7, %xmm6 # xmm6 = (xmm6 * mem) + xmm7
	addq	$8, %rdi
	vmovsd	%xmm6, (%rdx,%r14,8)
	cmpq	%rdi, %rax
	jne	.LBB0_19
	jmp	.LBB0_20
	.p2align	4, 0x90
.LBB0_10:                               #   in Loop: Header=BB0_6 Depth=2
	vxorpd	%xmm6, %xmm6, %xmm6
	vxorpd	%xmm7, %xmm7, %xmm7
	vxorpd	%xmm8, %xmm8, %xmm8
	vxorpd	%xmm10, %xmm10, %xmm10
	xorl	%r8d, %r8d
	vmovdqa64	%zmm1, %zmm9
	.p2align	4, 0x90
.LBB0_11:                               #   Parent Loop BB0_3 Depth=1
                                        #     Parent Loop BB0_6 Depth=2
                                        # =>    This Inner Loop Header: Depth=3
	vpaddq	%zmm2, %zmm9, %zmm11
	vpmullq	%zmm0, %zmm9, %zmm14
	vpaddq	%zmm3, %zmm9, %zmm12
	vpaddq	%zmm4, %zmm9, %zmm13
	vxorpd	%xmm15, %xmm15, %xmm15
	kxnorw	%k0, %k0, %k1
	vpaddq	%zmm5, %zmm9, %zmm9
	vpmullq	%zmm0, %zmm11, %zmm11
	vpmullq	%zmm0, %zmm12, %zmm12
	vpmullq	%zmm0, %zmm13, %zmm13
	vgatherqpd	(%rdi,%zmm14,8), %zmm15 {%k1}
	vxorpd	%xmm14, %xmm14, %xmm14
	kxnorw	%k0, %k0, %k1
	vfmadd231pd	-192(%r10,%r8), %zmm15, %zmm6 # zmm6 = (zmm15 * mem) + zmm6
	vgatherqpd	(%rdi,%zmm11,8), %zmm14 {%k1}
	vxorpd	%xmm11, %xmm11, %xmm11
	kxnorw	%k0, %k0, %k1
	vfmadd231pd	-128(%r10,%r8), %zmm14, %zmm7 # zmm7 = (zmm14 * mem) + zmm7
	vgatherqpd	(%rdi,%zmm12,8), %zmm11 {%k1}
	vxorpd	%xmm12, %xmm12, %xmm12
	kxnorw	%k0, %k0, %k1
	vfmadd231pd	-64(%r10,%r8), %zmm11, %zmm8 # zmm8 = (zmm11 * mem) + zmm8
	vgatherqpd	(%rdi,%zmm13,8), %zmm12 {%k1}
	vfmadd231pd	(%r10,%r8), %zmm12, %zmm10 # zmm10 = (zmm12 * mem) + zmm10
	addq	$256, %r8                       # imm = 0x100
	cmpq	%r8, %rbp
	jne	.LBB0_11
# %bb.12:                               #   in Loop: Header=BB0_6 Depth=2
	vaddpd	%zmm6, %zmm7, %zmm6
	vaddpd	%zmm8, %zmm10, %zmm8
	movq	-96(%rsp), %rcx                 # 8-byte Reload
	vaddpd	%zmm6, %zmm8, %zmm6
	movq	%rcx, %r8
	vextractf64x4	$1, %zmm6, %ymm7
	vaddpd	%zmm7, %zmm6, %zmm6
	vextractf128	$1, %ymm6, %xmm7
	vaddpd	%xmm7, %xmm6, %xmm6
	vshufpd	$1, %xmm6, %xmm6, %xmm7         # xmm7 = xmm6[1,0]
	vaddsd	%xmm7, %xmm6, %xmm6
	vmovsd	%xmm6, (%rdx,%r14,8)
	cmpq	%rax, %rcx
	jne	.LBB0_13
	.p2align	4, 0x90
.LBB0_20:                               #   in Loop: Header=BB0_6 Depth=2
	movq	%r14, %rdi
	orq	$1, %rdi
	vxorpd	%xmm6, %xmm6, %xmm6
	movq	$0, (%rdx,%rdi,8)
	cmpl	$32, %r15d
	jb	.LBB0_21
# %bb.22:                               #   in Loop: Header=BB0_6 Depth=2
	movq	-88(%rsp), %rcx                 # 8-byte Reload
	movq	-120(%rsp), %r8                 # 8-byte Reload
	addq	%rdi, %rcx
	leaq	(%r8,%rdi,8), %r12
	leaq	(%r8,%rcx,8), %rcx
	cmpq	%rcx, %r12
	movq	%rcx, %r8
	cmovaq	%r12, %r8
	cmovbq	%r12, %rcx
	addq	$8, %r8
	cmpq	%r8, -80(%rsp)                  # 8-byte Folded Reload
	setb	%r8b
	cmpq	-72(%rsp), %rcx                 # 8-byte Folded Reload
	setb	%cl
	andb	%r8b, %cl
	orb	-125(%rsp), %cl                 # 1-byte Folded Reload
	je	.LBB0_24
.LBB0_21:                               #   in Loop: Header=BB0_6 Depth=2
	xorl	%r8d, %r8d
.LBB0_27:                               #   in Loop: Header=BB0_6 Depth=2
	testb	$7, %al
	je	.LBB0_31
# %bb.28:                               #   in Loop: Header=BB0_6 Depth=2
	movq	%r11, %r15
	imulq	%r8, %r15
	addq	-112(%rsp), %r15                # 8-byte Folded Reload
	movq	-64(%rsp), %rcx                 # 8-byte Reload
	movq	%r8, %r12
	.p2align	4, 0x90
.LBB0_29:                               #   Parent Loop BB0_3 Depth=1
                                        #     Parent Loop BB0_6 Depth=2
                                        # =>    This Inner Loop Header: Depth=3
	vmovsd	(%r15), %xmm7                   # xmm7 = mem[0],zero
	addq	%r11, %r15
	vfmadd231sd	(%r9,%r12,8), %xmm7, %xmm6 # xmm6 = (xmm7 * mem) + xmm6
	incq	%r12
	decq	%rcx
	vmovsd	%xmm6, (%rdx,%rdi,8)
	jne	.LBB0_29
# %bb.30:                               #   in Loop: Header=BB0_6 Depth=2
	movq	-104(%rsp), %rcx                # 8-byte Reload
	subq	%r8, %rcx
	cmpq	$7, %rcx
	jb	.LBB0_5
	jmp	.LBB0_32
	.p2align	4, 0x90
.LBB0_31:                               #   in Loop: Header=BB0_6 Depth=2
	movq	%r8, %r12
	movq	-104(%rsp), %rcx                # 8-byte Reload
	subq	%r8, %rcx
	cmpq	$7, %rcx
	jb	.LBB0_5
.LBB0_32:                               #   in Loop: Header=BB0_6 Depth=2
	movq	%r11, %r8
	imulq	%r12, %r8
	addq	-112(%rsp), %r8                 # 8-byte Folded Reload
	.p2align	4, 0x90
.LBB0_33:                               #   Parent Loop BB0_3 Depth=1
                                        #     Parent Loop BB0_6 Depth=2
                                        # =>    This Inner Loop Header: Depth=3
	vmovsd	(%r8), %xmm7                    # xmm7 = mem[0],zero
	leaq	(%r8,%rax,8), %r15
	vfmadd132sd	-56(%r13,%r12,8), %xmm6, %xmm7 # xmm7 = (xmm7 * mem) + xmm6
	vmovsd	%xmm7, (%rdx,%rdi,8)
	vmovsd	(%r8,%rax,8), %xmm6             # xmm6 = mem[0],zero
	addq	%rbx, %r8
	vfmadd132sd	-48(%r13,%r12,8), %xmm7, %xmm6 # xmm6 = (xmm6 * mem) + xmm7
	vmovsd	%xmm6, (%rdx,%rdi,8)
	vmovsd	(%r11,%r15), %xmm7              # xmm7 = mem[0],zero
	addq	%r11, %r15
	vfmadd132sd	-40(%r13,%r12,8), %xmm6, %xmm7 # xmm7 = (xmm7 * mem) + xmm6
	vmovsd	%xmm7, (%rdx,%rdi,8)
	vmovsd	(%r11,%r15), %xmm6              # xmm6 = mem[0],zero
	addq	%r11, %r15
	vfmadd132sd	-32(%r13,%r12,8), %xmm7, %xmm6 # xmm6 = (xmm6 * mem) + xmm7
	vmovsd	%xmm6, (%rdx,%rdi,8)
	vmovsd	(%r11,%r15), %xmm7              # xmm7 = mem[0],zero
	addq	%r11, %r15
	vfmadd132sd	-24(%r13,%r12,8), %xmm6, %xmm7 # xmm7 = (xmm7 * mem) + xmm6
	vmovsd	%xmm7, (%rdx,%rdi,8)
	vmovsd	(%r11,%r15), %xmm6              # xmm6 = mem[0],zero
	addq	%r11, %r15
	vfmadd132sd	-16(%r13,%r12,8), %xmm7, %xmm6 # xmm6 = (xmm6 * mem) + xmm7
	vmovsd	%xmm6, (%rdx,%rdi,8)
	vmovsd	(%r11,%r15), %xmm7              # xmm7 = mem[0],zero
	addq	%r11, %r15
	vfmadd132sd	-8(%r13,%r12,8), %xmm6, %xmm7 # xmm7 = (xmm7 * mem) + xmm6
	vmovsd	%xmm7, (%rdx,%rdi,8)
	vmovsd	(%r11,%r15), %xmm6              # xmm6 = mem[0],zero
	vfmadd132sd	(%r13,%r12,8), %xmm7, %xmm6 # xmm6 = (xmm6 * mem) + xmm7
	addq	$8, %r12
	vmovsd	%xmm6, (%rdx,%rdi,8)
	cmpq	%r12, %rax
	jne	.LBB0_33
	jmp	.LBB0_5
	.p2align	4, 0x90
.LBB0_24:                               #   in Loop: Header=BB0_6 Depth=2
	vxorpd	%xmm6, %xmm6, %xmm6
	vxorpd	%xmm7, %xmm7, %xmm7
	vxorpd	%xmm8, %xmm8, %xmm8
	vxorpd	%xmm10, %xmm10, %xmm10
	xorl	%r8d, %r8d
	vmovdqa64	%zmm1, %zmm9
	.p2align	4, 0x90
.LBB0_25:                               #   Parent Loop BB0_3 Depth=1
                                        #     Parent Loop BB0_6 Depth=2
                                        # =>    This Inner Loop Header: Depth=3
	vpaddq	%zmm2, %zmm9, %zmm11
	vpmullq	%zmm0, %zmm9, %zmm14
	vpaddq	%zmm3, %zmm9, %zmm12
	vpaddq	%zmm4, %zmm9, %zmm13
	vxorpd	%xmm15, %xmm15, %xmm15
	kxnorw	%k0, %k0, %k1
	vpaddq	%zmm5, %zmm9, %zmm9
	vpmullq	%zmm0, %zmm11, %zmm11
	vpmullq	%zmm0, %zmm12, %zmm12
	vpmullq	%zmm0, %zmm13, %zmm13
	vgatherqpd	(%r12,%zmm14,8), %zmm15 {%k1}
	vxorpd	%xmm14, %xmm14, %xmm14
	kxnorw	%k0, %k0, %k1
	vfmadd231pd	-192(%r10,%r8), %zmm15, %zmm6 # zmm6 = (zmm15 * mem) + zmm6
	vgatherqpd	(%r12,%zmm11,8), %zmm14 {%k1}
	vxorpd	%xmm11, %xmm11, %xmm11
	kxnorw	%k0, %k0, %k1
	vfmadd231pd	-128(%r10,%r8), %zmm14, %zmm7 # zmm7 = (zmm14 * mem) + zmm7
	vgatherqpd	(%r12,%zmm12,8), %zmm11 {%k1}
	vxorpd	%xmm12, %xmm12, %xmm12
	kxnorw	%k0, %k0, %k1
	vfmadd231pd	-64(%r10,%r8), %zmm11, %zmm8 # zmm8 = (zmm11 * mem) + zmm8
	vgatherqpd	(%r12,%zmm13,8), %zmm12 {%k1}
	vfmadd231pd	(%r10,%r8), %zmm12, %zmm10 # zmm10 = (zmm12 * mem) + zmm10
	addq	$256, %r8                       # imm = 0x100
	cmpq	%r8, %rbp
	jne	.LBB0_25
# %bb.26:                               #   in Loop: Header=BB0_6 Depth=2
	vaddpd	%zmm6, %zmm7, %zmm6
	vaddpd	%zmm8, %zmm10, %zmm8
	movq	-96(%rsp), %rcx                 # 8-byte Reload
	vaddpd	%zmm6, %zmm8, %zmm6
	movq	%rcx, %r8
	vextractf64x4	$1, %zmm6, %ymm7
	vaddpd	%zmm7, %zmm6, %zmm6
	vextractf128	$1, %ymm6, %xmm7
	vaddpd	%xmm7, %xmm6, %xmm6
	vshufpd	$1, %xmm6, %xmm6, %xmm7         # xmm7 = xmm6[1,0]
	vaddsd	%xmm7, %xmm6, %xmm6
	vmovsd	%xmm6, (%rdx,%rdi,8)
	cmpq	%rax, %rcx
	je	.LBB0_5
	jmp	.LBB0_27
	.p2align	4, 0x90
.LBB0_34:                               #   in Loop: Header=BB0_3 Depth=1
	xorl	%r14d, %r14d
.LBB0_35:                               #   in Loop: Header=BB0_3 Depth=1
	testb	$1, %al
	je	.LBB0_2
# %bb.36:                               #   in Loop: Header=BB0_3 Depth=1
	vxorpd	%xmm6, %xmm6, %xmm6
	movq	$0, (%rdx,%r14,8)
	cmpl	$32, %r15d
	jae	.LBB0_42
.LBB0_37:                               #   in Loop: Header=BB0_3 Depth=1
	xorl	%edi, %edi
.LBB0_38:                               #   in Loop: Header=BB0_3 Depth=1
	testb	$7, %al
	je	.LBB0_47
# %bb.39:                               #   in Loop: Header=BB0_3 Depth=1
	movq	%r11, %rcx
	imulq	%rdi, %rcx
	movq	-64(%rsp), %r8                  # 8-byte Reload
	movq	%rdi, %rsi
	leaq	(%rcx,%r14,8), %rcx
	addq	-120(%rsp), %rcx                # 8-byte Folded Reload
	.p2align	4, 0x90
.LBB0_40:                               #   Parent Loop BB0_3 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	vmovsd	(%rcx), %xmm7                   # xmm7 = mem[0],zero
	addq	%r11, %rcx
	vfmadd231sd	(%r9,%rsi,8), %xmm7, %xmm6 # xmm6 = (xmm7 * mem) + xmm6
	incq	%rsi
	decq	%r8
	vmovsd	%xmm6, (%rdx,%r14,8)
	jne	.LBB0_40
# %bb.41:                               #   in Loop: Header=BB0_3 Depth=1
	movq	-104(%rsp), %rcx                # 8-byte Reload
	subq	%rdi, %rcx
	cmpq	$7, %rcx
	jb	.LBB0_2
	jmp	.LBB0_48
	.p2align	4, 0x90
.LBB0_42:                               #   in Loop: Header=BB0_3 Depth=1
	movq	-88(%rsp), %rcx                 # 8-byte Reload
	movq	-120(%rsp), %rdi                # 8-byte Reload
	movq	-80(%rsp), %r8                  # 8-byte Reload
	movq	-72(%rsp), %r12                 # 8-byte Reload
	addq	%r14, %rcx
	leaq	(%rdi,%r14,8), %rsi
	leaq	(%rdi,%rcx,8), %rdi
	cmpq	%rdi, %rsi
	movq	%rdi, %rcx
	cmovaq	%rsi, %rcx
	cmovbq	%rsi, %rdi
	addq	$8, %rcx
	cmpq	(%rsp), %r8                     # 8-byte Folded Reload
	setb	-112(%rsp)                      # 1-byte Folded Spill
	cmpq	%r12, 8(%rsp)                   # 8-byte Folded Reload
	setb	%r15b
	cmpq	%rcx, %r8
	setb	%cl
	cmpq	%r12, %rdi
	setb	%dil
	testb	%r15b, -112(%rsp)               # 1-byte Folded Reload
	jne	.LBB0_50
# %bb.43:                               #   in Loop: Header=BB0_3 Depth=1
	movl	-124(%rsp), %r15d               # 4-byte Reload
	andb	%dil, %cl
	movl	$0, %edi
	jne	.LBB0_38
# %bb.44:                               #   in Loop: Header=BB0_3 Depth=1
	vxorpd	%xmm6, %xmm6, %xmm6
	vxorpd	%xmm7, %xmm7, %xmm7
	vxorpd	%xmm8, %xmm8, %xmm8
	vxorpd	%xmm10, %xmm10, %xmm10
	xorl	%edi, %edi
	vmovdqa64	%zmm1, %zmm9
	.p2align	4, 0x90
.LBB0_45:                               #   Parent Loop BB0_3 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	vpaddq	%zmm2, %zmm9, %zmm11
	vpmullq	%zmm0, %zmm9, %zmm14
	vpaddq	%zmm3, %zmm9, %zmm12
	vpaddq	%zmm4, %zmm9, %zmm13
	vxorpd	%xmm15, %xmm15, %xmm15
	kxnorw	%k0, %k0, %k1
	vpaddq	%zmm5, %zmm9, %zmm9
	vpmullq	%zmm0, %zmm11, %zmm11
	vpmullq	%zmm0, %zmm12, %zmm12
	vpmullq	%zmm0, %zmm13, %zmm13
	vgatherqpd	(%rsi,%zmm14,8), %zmm15 {%k1}
	vxorpd	%xmm14, %xmm14, %xmm14
	kxnorw	%k0, %k0, %k1
	vfmadd231pd	-192(%r10,%rdi), %zmm15, %zmm6 # zmm6 = (zmm15 * mem) + zmm6
	vgatherqpd	(%rsi,%zmm11,8), %zmm14 {%k1}
	vxorpd	%xmm11, %xmm11, %xmm11
	kxnorw	%k0, %k0, %k1
	vfmadd231pd	-128(%r10,%rdi), %zmm14, %zmm7 # zmm7 = (zmm14 * mem) + zmm7
	vgatherqpd	(%rsi,%zmm12,8), %zmm11 {%k1}
	vxorpd	%xmm12, %xmm12, %xmm12
	kxnorw	%k0, %k0, %k1
	vfmadd231pd	-64(%r10,%rdi), %zmm11, %zmm8 # zmm8 = (zmm11 * mem) + zmm8
	vgatherqpd	(%rsi,%zmm13,8), %zmm12 {%k1}
	vfmadd231pd	(%r10,%rdi), %zmm12, %zmm10 # zmm10 = (zmm12 * mem) + zmm10
	addq	$256, %rdi                      # imm = 0x100
	cmpq	%rdi, %rbp
	jne	.LBB0_45
# %bb.46:                               #   in Loop: Header=BB0_3 Depth=1
	vaddpd	%zmm6, %zmm7, %zmm6
	vaddpd	%zmm8, %zmm10, %zmm8
	movq	-96(%rsp), %rcx                 # 8-byte Reload
	vaddpd	%zmm6, %zmm8, %zmm6
	movq	%rcx, %rdi
	vextractf64x4	$1, %zmm6, %ymm7
	vaddpd	%zmm7, %zmm6, %zmm6
	vextractf128	$1, %ymm6, %xmm7
	vaddpd	%xmm7, %xmm6, %xmm6
	vshufpd	$1, %xmm6, %xmm6, %xmm7         # xmm7 = xmm6[1,0]
	vaddsd	%xmm7, %xmm6, %xmm6
	vmovsd	%xmm6, (%rdx,%r14,8)
	cmpq	%rax, %rcx
	je	.LBB0_2
	jmp	.LBB0_38
.LBB0_47:                               #   in Loop: Header=BB0_3 Depth=1
	movq	%rdi, %rsi
	movq	-104(%rsp), %rcx                # 8-byte Reload
	subq	%rdi, %rcx
	cmpq	$7, %rcx
	jb	.LBB0_2
.LBB0_48:                               #   in Loop: Header=BB0_3 Depth=1
	movq	%r11, %rcx
	imulq	%rsi, %rcx
	leaq	(%rcx,%r14,8), %rdi
	addq	-120(%rsp), %rdi                # 8-byte Folded Reload
	.p2align	4, 0x90
.LBB0_49:                               #   Parent Loop BB0_3 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	vmovsd	(%rdi), %xmm7                   # xmm7 = mem[0],zero
	leaq	(%rdi,%rax,8), %rcx
	vfmadd132sd	-56(%r13,%rsi,8), %xmm6, %xmm7 # xmm7 = (xmm7 * mem) + xmm6
	vmovsd	%xmm7, (%rdx,%r14,8)
	vmovsd	(%rdi,%rax,8), %xmm6            # xmm6 = mem[0],zero
	addq	%rbx, %rdi
	vfmadd132sd	-48(%r13,%rsi,8), %xmm7, %xmm6 # xmm6 = (xmm6 * mem) + xmm7
	vmovsd	%xmm6, (%rdx,%r14,8)
	vmovsd	(%r11,%rcx), %xmm7              # xmm7 = mem[0],zero
	addq	%r11, %rcx
	vfmadd132sd	-40(%r13,%rsi,8), %xmm6, %xmm7 # xmm7 = (xmm7 * mem) + xmm6
	vmovsd	%xmm7, (%rdx,%r14,8)
	vmovsd	(%r11,%rcx), %xmm6              # xmm6 = mem[0],zero
	addq	%r11, %rcx
	vfmadd132sd	-32(%r13,%rsi,8), %xmm7, %xmm6 # xmm6 = (xmm6 * mem) + xmm7
	vmovsd	%xmm6, (%rdx,%r14,8)
	vmovsd	(%r11,%rcx), %xmm7              # xmm7 = mem[0],zero
	addq	%r11, %rcx
	vfmadd132sd	-24(%r13,%rsi,8), %xmm6, %xmm7 # xmm7 = (xmm7 * mem) + xmm6
	vmovsd	%xmm7, (%rdx,%r14,8)
	vmovsd	(%r11,%rcx), %xmm6              # xmm6 = mem[0],zero
	addq	%r11, %rcx
	vfmadd132sd	-16(%r13,%rsi,8), %xmm7, %xmm6 # xmm6 = (xmm6 * mem) + xmm7
	vmovsd	%xmm6, (%rdx,%r14,8)
	vmovsd	(%r11,%rcx), %xmm7              # xmm7 = mem[0],zero
	addq	%r11, %rcx
	vfmadd132sd	-8(%r13,%rsi,8), %xmm6, %xmm7 # xmm7 = (xmm7 * mem) + xmm6
	vmovsd	%xmm7, (%rdx,%r14,8)
	vmovsd	(%r11,%rcx), %xmm6              # xmm6 = mem[0],zero
	vfmadd132sd	(%r13,%rsi,8), %xmm7, %xmm6 # xmm6 = (xmm6 * mem) + xmm7
	addq	$8, %rsi
	vmovsd	%xmm6, (%rdx,%r14,8)
	cmpq	%rsi, %rax
	jne	.LBB0_49
	jmp	.LBB0_2
.LBB0_50:                               #   in Loop: Header=BB0_3 Depth=1
	movl	-124(%rsp), %r15d               # 4-byte Reload
	jmp	.LBB0_37
.LBB0_51:
	vpxor	%xmm0, %xmm0, %xmm0
	addq	$40, %rsp
	.cfi_def_cfa_offset 56
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%r12
	.cfi_def_cfa_offset 40
	popq	%r13
	.cfi_def_cfa_offset 32
	popq	%r14
	.cfi_def_cfa_offset 24
	popq	%r15
	.cfi_def_cfa_offset 16
	popq	%rbp
	.cfi_def_cfa_offset 8
	vzeroupper
	retq
.Lfunc_end0:
	.size	matrix_multiply_asm, .Lfunc_end0-matrix_multiply_asm
	.cfi_endproc
                                        # -- End function
	.ident	"Ubuntu clang version 18.1.3 (1ubuntu1)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
