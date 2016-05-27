/*
   +----------------------------------------------------------------------+
   | HipHop for PHP                                                       |
   +----------------------------------------------------------------------+
   | Copyright (c) 2010-2013 Facebook, Inc. (http://www.facebook.com)     |
   +----------------------------------------------------------------------+
   | This source file is subject to version 3.01 of the PHP license,      |
   | that is bundled with this package in the file LICENSE, and is        |
   | available through the world-wide-web at the following url:           |
   | http://www.php.net/license/3_01.txt                                  |
   | If you did not receive a copy of the PHP license and are unable to   |
   | obtain it through the world-wide-web, please send a note to          |
   | license@php.net so we can mail you a copy immediately.               |
   +----------------------------------------------------------------------+
*/

#include "hphp/runtime/vm/jit/vasm-emit.h"

#include "hphp/runtime/vm/jit/abi-arm.h"
#include "hphp/runtime/vm/jit/ir-instruction.h"
#include "hphp/runtime/vm/jit/mc-generator.h"
#include "hphp/runtime/vm/jit/print.h"
#include "hphp/runtime/vm/jit/reg-algorithms.h"
#include "hphp/runtime/vm/jit/service-requests.h"
#include "hphp/runtime/vm/jit/smashable-instr-arm.h"
#include "hphp/runtime/vm/jit/timer.h"
#include "hphp/runtime/vm/jit/vasm.h"
#include "hphp/runtime/vm/jit/vasm-gen.h"
#include "hphp/runtime/vm/jit/vasm-instr.h"
#include "hphp/runtime/vm/jit/vasm-internal.h"
#include "hphp/runtime/vm/jit/vasm-lower.h"
#include "hphp/runtime/vm/jit/vasm-print.h"
#include "hphp/runtime/vm/jit/vasm-reg.h"
#include "hphp/runtime/vm/jit/vasm-unit.h"
#include "hphp/runtime/vm/jit/vasm-util.h"
#include "hphp/runtime/vm/jit/vasm-visit.h"


#include "hphp/vixl/a64/macro-assembler-a64.h"

TRACE_SET_MOD(vasm);

namespace HPHP { namespace jit {
///////////////////////////////////////////////////////////////////////////////

using namespace arm;
using namespace vixl;

namespace arm { struct ImmFolder; }

namespace {
///////////////////////////////////////////////////////////////////////////////

const TCA kEndOfTargetChain = reinterpret_cast<TCA>(0xf00ffeeffaaff11f);

vixl::Register W(Vreg32 r) {
  PhysReg pr(r.asReg());
  return x2a(pr).W();
}

vixl::Register W(Vreg8 r) {
  PhysReg pr(r.asReg());
  return x2a(pr).W();
}

vixl::Register X(Vreg64 r) {
  PhysReg pr(r.asReg());
  return x2a(pr);
}

vixl::FPRegister S(Vreg r) {
  return x2simd(r).S();
}

vixl::FPRegister D(Vreg r) {
  return x2simd(r).D();
}

vixl::FPRegister Q(Vreg r) {
  return x2simd(r).Q();
}

// convert Vptr to MemOperand
vixl::MemOperand M(Vptr p) {
  assertx(p.base.isValid() && !p.index.isValid());
  return X(p.base)[p.disp];
}

vixl::Condition C(ConditionCode cc) {
  return convertCC(cc);
}


struct Vgen {
  explicit Vgen(Venv& env)
    : env(env)
    , text(env.text)
    , codeBlock(env.cb)
    , assem(*codeBlock)
    , a(&assem)
    , current(env.current)
    , next(env.next)
    , jmps(env.jmps)
    , jccs(env.jccs)
    , bccs(env.bccs)
    , catches(env.catches)
  {
    rtAsm = arm::rasm();
    rtVixl = arm::rvixl();
  }

  static void patch(Venv& env);
  static void pad(CodeBlock& cb);
  void emitSimdImmInt(int64_t val, Vreg d);

  /////////////////////////////////////////////////////////////////////////////

  template<class Inst> void emit(Inst& i) {
    always_assert_flog(false, "unimplemented instruction: {} in B{}\n",
                       vinst_names[Vinstr(i).op], size_t(current));
  }

  // intrinsics
  void emit(const copy& i);
  void emit(const copy2& i);
  void emit(const debugtrap& i) { a->Brk(0); }
  void emit(const fallthru& i) {}
  //void emit(const hostcall& i) { a->HostCall(i.argc); }
  void emit(const ldimmb& i);
  void emit(const ldimml& i);
  void emit(const ldimmq& i);
  void emit(const ldimmqs& i) { not_implemented(); }
  void emit(const load& i);
  void emit(const store& i);
  void emit(const mcprep& i);

  // functions
  void emit(const call& i);
  void emit(const calldestr& i);
  void emit(const callm& m);
  void emit(const callr& i);
  void emit(const calls& i);
  void emit(const ret& i) { a->Ret(); }

  // stub function abi
  void emit(const stublogue& i);
  void emit(const stubret& i);
  void emit(const callstub& i);
  void emit(const callfaststub& i);
  void emit(const tailcallstub& i);

  // php function abi
  void emit(const phplogue& i);
  void emit(const phpret& i);
  void emit(const tailcallphp& i);
  void emit(const callphp& i);
  void emit(const callarray& i);
  void emit(const contenter& i);
 
  // vm entry abi
  void emit(const calltc&);
  void emit(const leavetc& i);
  void emit(const resumetc&);

  // exceptions
  void emit(const landingpad& i) {a->Add(rSp, rSp, VASM_ARM_CALL_SP_OFF);}
  void emit(const nothrow& i);
  void emit(const syncpoint& i);
  void emit(const unwind& i);

  // instructions
  void emit(const andbi& i);
  void emit(const andq& i) { a->And(X(i.d), X(i.s1), X(i.s0)); }
  void emit(const andqi& i) { a->And(X(i.d), X(i.s1), i.s0.l()); }
  void emit(const andl& i) { a->And(W(i.d), W(i.s1), W(i.s0)); }
  void emit(const andli& i) { a->And(W(i.d), W(i.s1), i.s0.l()); }
  void emit(const addl& i) { a->Add(W(i.d), W(i.s1), W(i.s0), SetFlags); }
  void emit(const addli& i) { a->Add(W(i.d), W(i.s1), i.s0.l(), SetFlags); }
  void emit(const addlim& i);
  void emit(const addlm& i);
  void emit(const addq& i) { a->Add(X(i.d), X(i.s1), X(i.s0), SetFlags); }
  void emit(const addqi& i);
  void emit(const addsd& i) { a->Fadd(D(i.d), D(i.s1), D(i.s0)); }
  void emit(const brk& i) { a->Brk(i.code); }
  void emit(const cbcc &i);
  void emit(const cloadq& i);
  void emit(const cmovq& i) { a->Csel(X(i.d), X(i.t), X(i.f), C(i.cc)); }
  void emit(const cmovb& i) { a->Csel(W(Vreg32(size_t(i.d))), W(Vreg32(size_t(i.t))), W(Vreg32(size_t(i.f))), C(i.cc)); }
  void emit(const cmpb& i);
  void emit(const cmpbi& i);
  void emit(const cmpbim& i);
  void emit(const cmpl& i) { a->Cmp(W(i.s1), W(i.s0)); }
  void emit(const cmplm& i);
  void emit(const cmpli& i) { a->Cmp(W(i.s1), i.s0.l()); }
  void emit(const cmplib& i) { a->Cmp(W(i.s1), i.s0.b()); }
  void emit(const cmplim& i);
  void emit(const cmpq& i) { a->Cmp(X(i.s1), X(i.s0)); }
  void emit(const cmpqi& i) { a->Cmp(X(i.s1), i.s0.l()); }
  void emit(const cmpqib& i) { a->Cmp(X(i.s1), i.s0.b()); }
  void emit(const cmpqim& i);
  void emit(const cmpqiw& i) { a->Cmp(X(i.s1), i.s0.w()); }
  void emit(const cmpwim& i);
  void emit(const cmpqm& i);
  void emit(const cvtsi2sd& i) {  a->Scvtf(D(i.d), X(i.s)); };
  void emit(const decl& i) { a->Sub(W(i.d), W(i.s), 1, SetFlags); }
  void emit(const declm& i);
  void emit(const decq& i) { a->Sub(X(i.d), X(i.s), 1LL, SetFlags); }
  void emit(const decqm& i);
  void emit(const divint& i) { a->Sdiv(X(i.d), X(i.s0), X(i.s1)); }
  void emit(const divsd& i) { a->Fdiv(D(i.d), D(i.s1), D(i.s0)); }
  void emit(const divsq& i) { a->Sdiv(X(i.d), X(i.s0), X(i.s1)); }
  void emit(const fabs& i) { a->Fabs(D(i.d), D(i.s)); }
  void emit(const incl& i) { a->Add(W(i.d), W(i.s), 1, SetFlags); }
  void emit(const inclm& i);
  void emit(const incq& i) { a->Add(X(i.d), X(i.s), 1LL, SetFlags); }
  void emit(const incqexl& i);
  void emit(const incqm& i);
  void emit(const incwm& i);
  void emit(const imul& i); 
  void emit(const jcc& i);
  void emit(const jcci& i);
  void emit(const jmp& i);
  void emit(const jmpi& i);
  void emit(const jmpm& i);
  void emit(const jmpr& i) { a->Br(X(i.target)); }
  void emit(const lea& i);
  void emit(const leap& i);
  void emit(const ldimmqa& i);
  void emit(const loadb& i);
  void emit(const loadexl& i) { a->Ldxr(X(i.d), M(i.s)); }
  void emit(const loadtqb& i);
  void emit(const loadl& i) { a->Ldr(W(i.d), prepMem(i.s)); }
  void emit(const loadpq& i) { a->Ldp(X(i.d1), X(i.d2), prepMem(i.s)); }
  void emit(const loadqp& i);
  void emit(const loadsbq& i) { a->Ldrsb(X(i.d), prepMem(i.s)); }//Sbit extends
  void emit(const loadslq& i) { a->Ldrsw(X(i.d), prepMem(i.s)); }//Sbit extends
  void emit(const loadswq& i) { a->Ldrsh(X(i.d), prepMem(i.s)); }//Sbit extends
  void emit(const loadups& i);
  void emit(const loadzbl& i) { a->Ldrb(W(i.d), prepMem(i.s)); } // 0 extends
  void emit(const loadzlq& i);
  void emit(const loadzbq& i) { a->Ldrb(X(i.d), prepMem(i.s));  }
  void emit(const movb& i) { a->Bfi(W(Vreg32(size_t(i.d))),W(Vreg32(size_t(i.s))), 0, 8); }
  void emit(const movl& i) { a->Mov(W(i.d), W(i.s));}
  void emit(const movsbl& i) { a->Sxtb(W(i.d), W(Vreg32(size_t(i.s)))); }
  void emit(const movtqb& i) { a->Bfi(X(Vreg64(size_t(i.d))), X(i.s), 0, 8);}
  void emit(const movtql& i) { a->Mov(W(i.d), W(Vreg32(size_t(i.s))));}
  void emit(const movzbl& i) { a->Uxtb(W(i.d), W(i.s)); }
  void emit(const movzbq& i) { a->Uxtb(X(i.d), X(Vreg64(size_t(i.s)))); }
  void emit(const movzlq& i) { a->Uxtw(X(i.d), X(Vreg64(size_t(i.s)))); }
  void emit(const mulsd& i) { a->Fmul(D(i.d), D(i.s1), D(i.s0)); }
  void emit(const neg& i) { a->Neg(X(i.d), X(i.s), vixl::SetFlags); }
  void emit(const not& i) { a->Mvn(X(i.d), X(i.s)); }
  void emit(const orq& i) { a->Orr(X(i.d), X(i.s1), X(i.s0)); }
  void emit(const orqi& i) { a->Orr(X(i.d), X(i.s1), i.s0.l()); }
  void emit(const orwim& i);
  void emit(const pop& i);
  void emit(const popm& i);
  void emit(const push& i);
  void emit(const sar& i) { a->Asr(X(i.d), X(i.s1), X(i.s0)); }
  void emit(const sarqi& i) { a->Asr(X(i.d), X(i.s1), i.s0.l()); }
  void emit(const shl& i) { a->Lsl(X(i.d), X(i.s1), X(i.s0)); }
  void emit(const shlli& i) { a->Lsl(W(i.d), W(i.s1), i.s0.l()); }
  void emit(const shlqi& i);
  void emit(const shrli& i) { a->Lsr(W(i.d), W(i.s1), i.s0.l()); }
  void emit(const shrqi& i);
  void emit(const storeb& i) { a->Strb(W(i.s), prepMem(i.m)); }
  void emit(const storebi& i);
  void emit(const storeexl& i) { a->Stxr(X(i.st), X(i.s), M(i.d)); }
  void emit(const storeli& i);
  void emit(const storeqi& i);
  void emit(const storel& i) { a->Str(W(i.s), prepMem(i.m));  }
  void emit(const storew& i) { a->Strh(W(Vreg32(size_t(i.s))), prepMem(i.m));}
  void emit(const setcc& i) { PhysReg r(i.d.asReg()); a->Cset(X(r), C(i.cc));}
  void emit(const srem& i);
  void emit(const subli& i) { a->Sub(W(i.d), W(i.s1), i.s0.l(), SetFlags);}
  void emit(const subq& i) { a->Sub(X(i.d), X(i.s1), X(i.s0), SetFlags);}
  void emit(const subsd& i) { a->Fsub(D(i.d), D(i.s1), D(i.s0)); }
  void emit(const subqi& i);
  void emit(const subbi& i) { a->Sub(W(i.d), W(i.s1), i.s0.b(), SetFlags); }
  void emit(const storeups& i);
  void emit(const tbcc& i);
  void emit(const testl& i) { a->Tst(W(i.s1), W(i.s0)); }
  void emit(const testli& i) { a->Tst(W(i.s1), i.s0.l()); }
  void emit(const testq& i) { a->Tst(X(i.s1), X(i.s0)); }
  void emit(const testbi& i) { a->Tst(W(i.s1), i.s0.b()); }
  void emit(const testqib& i) { a->Tst(X(i.s1), i.s0.b()); }
  void emit(const testqiw& i) { a->Tst(X(i.s1), i.s0.w()); }
  void emit(const testbim& i);
  void emit(const testlim& i);
  void emit(const testqim& i);
  void emit(const testwim& i);
  void emit(const testqm& i);
  void emit(const testb& i);
  void emit(const testqi& i) { a->Tst(X(i.s1), i.s0.l()); }
  void emit(const ud2& i) { a->Brk(1); }
  void emit(const ucomisd& i) { a->Fcmp(D(i.s0), D(i.s1)); }
  void emit(const xorb& i) { a->Eor(W(i.d), W(i.s1), W(i.s0)); }
  void emit(const xorq& i) { a->Eor(X(i.d), X(i.s1), X(i.s0)); }
  void emit(const xorqi& i) { a->Eor(X(i.d), X(i.s1), i.s0.l()); }
  void emit(const xorl& i) { a->Eor(W(i.d), W(i.s1), W(i.s0)); }
  void emit(const xorli& i) { a->Eor(W(i.d), W(i.s1), i.s0.l()); }

  // ARM specific instructions
  void emit(const cpfpsr& i) {a->Mrs(X(i.d), FPSR);}
  void emit(const setfpsr& i) {a->Msr(FPSR, X(i.s));}
  void emit(const fcvtzs& i) {a->Fcvtzs(X(i.d), D(i.s));}

private:
  //Convert Vptr to MemOperand for load/store
  vixl::MemOperand prepMem(Vptr m) {
    static const int scaleTbl[] = { -1, 0, 1, -1, 2, -1, -1, -1, 3 };
    if(m.base.isValid() && m.index.isValid()) {
       int scale=scaleTbl[m.scale];
       assert(scale != -1);
       a->Add(X(rtAsm), X(m.base), m.disp, SetFlags);
       return MemOperand{X(rtAsm), X(m.index), vixl::LSL, (unsigned)scale};
    } else if(m.base.isValid()) {
       return X(m.base)[m.disp];
    } else if (m.index.isValid()) {
       int scale=scaleTbl[m.scale];
       assert(scale != -1);
       a->Mov(X(rtAsm), m.disp);
       return MemOperand{X(rtAsm), X(m.index), vixl::LSL, (unsigned)scale};
    } else {
       return vixl::xzr[m.disp];
    }
  }

  //Derive the target address from the input Vptr
  void prepAddr(Vptr m, Vreg64 d) {
    static const int scaleTbl[] = { -1, 0, 1, -1, 2, -1, -1, -1, 3 };

    if (m.base.isValid() && m.index.isValid()) {
      int scale = scaleTbl[m.scale];
      assertx(scale != -1);
      // dest = base + scale*index + displacement
      a->Add(X(rtAsm), X(m.base), m.disp);
      Operand opr{X(m.index), vixl::LSL, (unsigned)scale};
      a->Add(X(d), X(rtAsm), opr, LeaveFlags);
    } else if(m.base.isValid()) {
      a->Add(X(d), X(m.base), m.disp);
    } else if (m.index.isValid()) {
      int scale = scaleTbl[m.scale];
      assertx(scale != -1);
      a->Mov(X(rtAsm), m.disp);
      Operand opr{X(m.index), vixl::LSL, (unsigned)scale};
      a->Add(X(d), X(rtAsm), opr, LeaveFlags);
    } else  {
      //baseless
      a->Mov(X(d), m.disp);
    }
  }

  CodeBlock& frozen() { return text.frozen().code; }

private:
  Venv& env;
  Vtext& text;
  CodeBlock* codeBlock;
  vixl::MacroAssembler assem;
  vixl::MacroAssembler* a;

  const Vlabel current;
  const Vlabel next;
  jit::vector<Venv::LabelPatch>& jmps;
  jit::vector<Venv::LabelPatch>& jccs;
  jit::vector<Venv::LabelPatch>& bccs;
  jit::vector<Venv::LabelPatch>& catches;

  Vreg rtAsm;
  Vreg rtVixl;
};

///////////////////////////////////////////////////////////////////////////////

void Vgen::patch(Venv& env) {
  for (auto& p : env.jmps) {
    assertx(env.addrs[p.target]);
    smashJmp(p.instr, env.addrs[p.target]);
  }
  for (auto& p : env.jccs) {
    assertx(env.addrs[p.target]);
    smashJcc(p.instr, env.addrs[p.target]);
  }
  for (auto& p : env.bccs) {
    assertx(env.addrs[p.target]);
    auto link = (Instruction*) p.instr;
    link->SetImmPCOffsetTarget(Instruction::Cast(env.addrs[p.target]));
  }
}

// This is required for relocation.
void Vgen::pad(CodeBlock& cb) {
  vixl::MacroAssembler a { cb };
  assertx(cb.available() >= ARM_INSTR_WIDTH);
  while (cb.available() >= ARM_INSTR_WIDTH)
   a.Brk(1);
}

void Vgen::emitSimdImmInt(int64_t val, Vreg d) {
  if (val == 0) {
    a->Fmov(D(d), vixl::xzr);
  } else {
    a->Mov(X(rtAsm), val);
    a->Fmov(D(d), X(rtAsm));
  }
}

///////////////////////////////////////////////////////////////////////////////

void Vgen::emit(const copy& i) {
  if (i.s.isGP() && i.d.isGP()) {
    a->Mov(X(i.d), X(i.s));
  } else if (i.s.isSIMD() && i.d.isGP()) {
    a->Fmov(X(i.d), D(i.s));
  } else if (i.s.isGP() && i.d.isSIMD()) {
    a->Fmov(D(i.d), X(i.s));
  } else {
    assertx(i.s.isSIMD() && i.d.isSIMD());
    a->Fmov(D(i.d), D(i.s));
  }
}

void Vgen::emit(const copy2& i) {
  assertx(i.s0.isValid() && i.s1.isValid() && i.d0.isValid() && i.d1.isValid());
  auto s0 = i.s0, s1 = i.s1, d0 = i.d0, d1 = i.d1;
  assertx(d0 != d1);
  if (d0 == s1) {
    if (d1 == s0) {
      a->Eor(X(d0), X(d0), X(d1));
      a->Eor(X(d1), X(d0), X(d1));
      a->Eor(X(d0), X(d0), X(d1));
    } else {
      // could do this in a simplify pass
      if (s1 != d1)  emit(copy{s1, d1}); // save s1 first; d1 != s0
      if (s0 != d0)  emit(copy{s0, d0});
    }
  } else {
    // could do this in a simplify pass
    if (s0 != d0) emit(copy{s0, d0});
    if (s1 != d1) emit(copy{s1, d1});
  }

}

void Vgen::emit(const ldimmb& i) {
  // ldimmb is for Vconst::Byte, which is treated as unsigned uint8_t
  auto val = i.s.ub();
  if (i.d.isSIMD()) {
    emitSimdImmInt(val, i.d);
  } else {
    Vreg32 d = i.d;
    a->Mov(W(d), val);
  }
}

void Vgen::emit(const ldimml& i) {
  // ldimml is for Vconst::Long, which is treated as unsigned uint32_t
  auto val = i.s.l();
  if (i.d.isSIMD()) {
    emitSimdImmInt((uint32_t)val, i.d);
  } else {
    Vreg32 d = i.d;
    a->Mov(W(d), val);
  }
}

void Vgen::emit(const ldimmq& i) {
  union { double dval; int64_t ival; };
  ival = i.s.q();
  if (i.d.isSIMD()) {
    // Assembler::fmov (which you'd think shouldn't be a macro instruction)
    // will emit a ldr from a literal pool if IsImmFP64 is false. vixl's
    // literal pools don't work well with our codegen pattern, so if that
    // would happen, emit the raw bits into a GPR first and then move them
    // unmodified into a SIMD.
    if (vixl::Assembler::IsImmFP64(dval)) {
      a->Fmov(D(i.d), dval);
    } else if (ival == 0) { // careful: dval==0.0 is true for -0.0
      // 0.0 is not encodeable as an immediate to Fmov, but this works.
      a->Fmov(D(i.d), vixl::xzr);
    } else {
      a->Mov(X(rtAsm), ival);
      a->Fmov(D(i.d), X(rtAsm));
    }
  } else {
    a->Mov(X(i.d), ival);
  }
}

void Vgen::emit(const load& i) {
  vixl::MemOperand m = prepMem(i.s);
  if (i.d.isGP()) {
    a->Ldr(X(i.d), m);
  } else {
    a->Ldr(D(i.d), m);
  }
}

void Vgen::emit(const store& i) {
  vixl::MemOperand m = prepMem(i.d);
  if (i.s.isGP()) {
    a->Str(X(i.s), m);
  } else {
    a->Str(D(i.s), m);
  }
}

///////////////////////////////////////////////////////////////////////////////

void Vgen::emit(const mcprep& i) {
  /*
  * Initially, we set the cache to hold (addr << 1) | 1 (where `addr' is the
  * address of the movq) so that we can find the movq from the handler.
  *
  * We set the low bit for two reasons: the Class* will never be a valid
  * Class, so we'll always miss the inline check before it's smashed, and
  * handlePrimeCacheInit can tell it's not been smashed yet
  */
  auto const mov_addr = emitSmashableMovq(*codeBlock, env.meta, 0, r64(i.d));
  auto const imm = reinterpret_cast<uint64_t>(mov_addr);
  smashMovq(mov_addr, (imm << 1) | 1);

  env.meta.addressImmediates.insert(reinterpret_cast<TCA>(~imm));
}

///////////////////////////////////////////////////////////////////////////////

void Vgen::emit(const call& i) {
  auto const b = ssize_t(i.target);
  emit(ldimmqa{b, rtAsm});
  emit(callr{rtAsm, i.args});
}

void Vgen::emit(const calldestr& i) {
  //it is baseless memory type
  prepAddr(i.target, rtAsm);
  a->Ldr(X(rtAsm), vixl::MemOperand(X(rtAsm)[0]));

  a->Stp(rFp, rLinkReg, vixl::MemOperand(rSp, -VASM_ARM_CALL_SP_OFF, vixl::PreIndex));
  a->Mov(rFp, rSp);
  a->Blr(X(rtAsm));
  a->Ldp(rFp, rLinkReg, vixl::MemOperand(rSp, VASM_ARM_CALL_SP_OFF, vixl::PostIndex));
}

void Vgen::emit(const callm& i) {
  prepAddr(i.target, rtAsm);
  a->Ldr(X(rtAsm), vixl::MemOperand(X(rtAsm)[0]));
  emit(callr{rtAsm});
}

void Vgen::emit(const callr& i) {
  a->Stp(vixl::xzr, rLinkReg,
         vixl::MemOperand(rSp, -VASM_ARM_CALL_SP_OFF, vixl::PreIndex));
  a->Blr(X(i.target));
  a->Ldp(vixl::xzr, rLinkReg,
         vixl::MemOperand(rSp, VASM_ARM_CALL_SP_OFF, vixl::PostIndex));
}

void Vgen::emit(const calls& i) {
  emitSmashableCall(*codeBlock, env.meta, i.target);
}

///////////////////////////////////////////////////////////////////////////////

/* entry and exit 2 : See tailcallstub below along with these */
void Vgen::emit(const stublogue& i) {
  a->Mov(rAsm, rSp);
  if (i.saveframe) {
    a->Stp(X(rvmfp()), rLinkReg,
           vixl::MemOperand(rAsm, -VASM_ARM_CALL_SP_OFF, vixl::PreIndex));
  } else {
    a->Stp(vixl::xzr, rLinkReg,
           vixl::MemOperand(rAsm, -VASM_ARM_CALL_SP_OFF, vixl::PreIndex));
  }
  a->Mov(rSp, rAsm);
}

void Vgen::emit(const stubret& i) {
  a->Mov(rAsm, rSp);
  if (i.saveframe) {
    a->Ldp(X(rvmfp()), rLinkReg,
           vixl::MemOperand(rAsm, VASM_ARM_CALL_SP_OFF, vixl::PostIndex));
  } else {
    a->Ldp(vixl::xzr, rLinkReg,
           vixl::MemOperand(rAsm, VASM_ARM_CALL_SP_OFF, vixl::PostIndex));
  }
  a->Mov(rSp, rAsm);
  a->Ret();
}

void Vgen::emit(const callstub& i) {
  auto const b = ssize_t(i.target);
  emit(ldimmq{b, rtAsm});
  emit(callr{rtAsm, i.args});
}

void Vgen::emit(const callfaststub& i) {
  auto const b = ssize_t(i.target);
  emit(ldimmq{b, rtAsm});
  emit(callr{rtAsm, i.args});
  emit(syncpoint{i.fix});
}

void Vgen::emit(const tailcallstub& i) {
  auto const b = ssize_t(i.target);
  a->Add(rSp, rSp, VASM_ARM_CALL_SP_OFF);
  emit(ldimmq{b, rtAsm});
  emit(jmpr{rtAsm, i.args});
}

///////////////////////////////////////////////////////////////////////////////

/* entry and exit 1 */
void Vgen::emit(const phplogue& i) {
  /* Save LR to FP->m_savedRIP */
  vixl::MemOperand m = prepMem(i.fp[AROFF(m_savedRip)]);
  a->Str(rLinkReg, m);
}

void Vgen::emit(const phpret& i) {
  vixl::MemOperand m = prepMem(i.fp[AROFF(m_savedRip)]);
  a->Ldr(rLinkReg, m);
  if(!i.noframe) {
    vixl::MemOperand m = prepMem(i.fp[AROFF(m_sfp)]);
    a->Ldr(X(i.d), m);
  }
  a->Ret();
}

void Vgen::emit(const tailcallphp& i) {
  /* Save current functions return address to LR */
  vixl::MemOperand m = prepMem(i.fp[AROFF(m_savedRip)]);
  a->Ldr(rLinkReg, m);
  a->Br(X(i.target));
}

void Vgen::emit(const callphp& i) {
  emitSmashableCall(*codeBlock, env.meta, i.stub);
  emit(unwind{{i.targets[0], i.targets[1]}});
}

void Vgen::emit(const callarray& i) {
  auto const b = ssize_t(i.target);
  emit(ldimmq{b, rtAsm});
  emit(callr{rtAsm, i.args});
}

void Vgen::emit(const calltc& i) {
  a->Mov(X(rtAsm), i.exittc);
  a->Stp(vixl::xzr, X(rtAsm), vixl::MemOperand(rSp, -16, vixl::PreIndex));
  vixl::MemOperand m = prepMem(i.fp[AROFF(m_savedRip)]);
  a->Ldr(rLinkReg, m);
  a->Br(X(i.target));
}

void Vgen::emit(const resumetc& i) {
  a->Mov(rLinkReg, i.exittc);
  emit(callr{i.target, i.args});
  a->Ret();
}

void Vgen::emit(const leavetc& i) {
  a->Mov(rAsm, rSp);
  a->Ldp(vixl::xzr, X(rtVixl), vixl::MemOperand(rAsm, 16, vixl::PostIndex));
  a->Mov(rSp, rAsm);
  a->Br(X(rtVixl));
}

void Vgen::emit(const contenter& i) {
  vixl::Label Stub, End;

  a->B(&End);

  a->bind(&Stub);
  a->Str(rLinkReg, M(i.fp[AROFF(m_savedRip)]));
  a->Br(X(i.target));

  a->bind(&End);

  a->Stp(vixl::xzr, rLinkReg,
         vixl::MemOperand(rSp, -VASM_ARM_CALL_SP_OFF, vixl::PreIndex));
  a->Bl(&Stub);
  a->Ldp(vixl::xzr, rLinkReg,
         vixl::MemOperand(rSp, VASM_ARM_CALL_SP_OFF, vixl::PostIndex));
  // m_savedRip will point here.
  emit(unwind{{i.targets[0], i.targets[1]}});
}

///////////////////////////////////////////////////////////////////////////////

void Vgen::emit(const nothrow& i) {
  env.meta.catches.emplace_back(a->frontier(), nullptr);
}

void Vgen::emit(const syncpoint& i) {
  int off = getCallLrOff(a->frontier());
  FTRACE(5, "IR recordSyncPoint: {} {} {}\n", a->frontier() - off,
         i.fix.pcOffset, i.fix.spOffset);
  env.meta.fixups.emplace_back(a->frontier() - off, i.fix);
}

void Vgen::emit(const unwind& i) {
  int off = getCallLrOff(a->frontier());
  catches.push_back({a->frontier() - off, i.targets[1]});
  emit(jmp{i.targets[0]});
}

///////////////////////////////////////////////////////////////////////////////

void Vgen::emit(const andbi& i) {
  emit(andli{i.s0,  Vreg32(size_t(i.s1)), rtAsm, i.sf});
  emit(movb{Vreg8(size_t(rtAsm)), i.d});
}

void Vgen::emit(const addlm& i) {
  // Cannot use rasm here as the conflict b/w prep(m) and store
  emit(loadl{i.m, rtVixl});
  emit(addl{i.s0, rtVixl, rtVixl, i.sf});
  emit(storel{rtVixl, i.m});
}

void Vgen::emit(const addlim& i) {
  // Cannot use rasm here as the conflict b/w prep(m) and store
  emit(loadl{i.m, rtVixl});
  emit(addli{i.s0, rtVixl, rtVixl, i.sf});
  emit(storel{rtVixl, i.m});
}

void Vgen::emit(const addqi& i) {
  if(X(i.d).IsSP()) {
    a->Add(X(i.d), X(i.s1), i.s0.l());
  } else {
    a->Add(X(i.d), X(i.s1), i.s0.l(), SetFlags);
  }
}

void Vgen::emit(const cbcc& i) {
  assertx(i.cc == vixl::ne || i.cc == vixl::eq);
  if (i.targets[1] != i.targets[0]) {
    if (next == i.targets[1]) {
      // the taken branch is the fall-through block, invert the branch.
      return emit(cbcc{i.cc == vixl::ne ? vixl::eq : vixl::ne, i.s,
               {i.targets[1], i.targets[0]}});
    }
    bccs.push_back({a->frontier(), i.targets[1]});
    // offset range +/- 1MB
    if (i.cc == vixl::ne) {
      a->cbnz(X(i.s), 0);
    } else {
      a->cbz(X(i.s), 0);
    }
  }
  emit(jmp{i.targets[0]});
}

void Vgen::emit(const cloadq& i) {
  emit(load{i.t, rtAsm});
  emit(cmovq{i.cc, i.sf, i.f, rtAsm, i.d});
}

void Vgen::emit(const cmpb& i) {
  emit(movsbl{i.s0, rtAsm});
  emit(movsbl{i.s1, rtVixl});
  emit(cmpl{rtAsm, rtVixl, i.sf});
}

void Vgen::emit(const cmpbi& i) {
  emit(movsbl{i.s1, rtAsm});
  emit(cmplib{i.s0, rtAsm, i.sf});
}

void Vgen::emit(const cmpbim& i) {
  // As immediate are int32_t and it is converted to int64_t
  // in vixl, we need sign extension operation here
  emit(loadsbq{i.s1, rtAsm});
  emit(cmpqib{i.s0, rtAsm, i.sf});
}

void Vgen::emit(const cmpqm& i) {
  emit(load{i.s1, rtAsm});
  emit(cmpq{i.s0, rtAsm, i.sf});
}

void Vgen::emit(const cmpqim& i) {
  emit(load{i.s1, rtAsm});
  emit(cmpqi{i.s0, rtAsm, i.sf});
}

void Vgen::emit(const cmplim& i) {
  // As immediate are int32_t and it is converted to int64_t
  // in vixl, we need sign extension operation here.
  // It is safe to use X(rtAsm) as src & dst, as prepMem(m) is not needed
  // after loadslq.
  emit(loadslq{i.s1, rtAsm});
  // As maximum immed is int32, we can use qi here.
  emit(cmpqi{i.s0, rtAsm, i.sf});
}

void Vgen::emit(const cmplm& i) {
  emit(loadl{i.s1, rtAsm});
  emit(cmpl{i.s0, rtAsm, i.sf});
}

void Vgen::emit(const cmpwim& i) {
  // As immediate are int32_t and it is converted to int64_t
  // in vixl, we need sign extension operation here
  emit(loadswq{i.s1, rtAsm});
  emit(cmpqiw{i.s0, rtAsm, i.sf});
}

void Vgen::emit(const declm& i) {
  // Cannot use rasm here as the conflict b/w prep(m) and store
  emit(loadl{i.m, rtVixl});
  emit(decl{rtVixl, rtVixl, i.sf});
  emit(storel{rtVixl, i.m});
}

void Vgen::emit(const decqm& i) {
  // Cannot use rasm here as the conflict b/w prep(m) and store
  emit(load{i.m, rtVixl});
  emit(decq{rtVixl, rtVixl, i.sf});
  emit(store{rtVixl, i.m});
}

void Vgen::emit(const inclm& i) {
  // Cannot use rasm here as the conflict b/w prep(m) and store
  emit(loadl{i.m, rtVixl});
  emit(incl{rtVixl, rtVixl, i.sf});
  emit(storel{rtVixl, i.m});
}

void Vgen::emit(const incqexl& i) {
  vixl::Label loop;

  a->bind(&loop);
  a->Ldxr(X(i.s0), M(i.m));
  a->Add(X(i.s0), X(i.s0), 1);
  a->Stxr(X(i.s1), X(i.s0), M(i.m));
  a->Cmp(X(i.s1), 1);
  a->B(&loop, vixl::eq);
}

void Vgen::emit(const incqm& i) {
  // Cannot use rasm here as the conflict b/w prep(m) and store
  emit(load{i.m, rtVixl});
  emit(incq{rtVixl, rtVixl, i.sf});
  emit(store{rtVixl, i.m});
}

void Vgen::emit(const incwm& i) {
  // Cannot use rasm here as the conflict b/w prep(m) and store
  emit(loadswq{i.m, rtVixl});
  emit(incq{rtVixl, rtVixl, i.sf});
  emit(storew{rtVixl, i.m});
}

void Vgen::emit(const imul& i) {
  vixl::Label End;
  // Mul instruction does not update the carry/overflow flag.
  // Need extra logic to set the overflow flag.
  // As this is a signed multiplication, there is no overflow if the sign bit
  // is same on both lo/hi registers.

  // clear the carry/overflow flag from the previous instruction
  a->Cmp(vixl::xzr, 0);
  // Get the uppert bits of the mulitplication
  a->Smulh(X(rtVixl), X(i.s0), X(i.s1));
  a->Mul(X(i.d), X(i.s0), X(i.s1));
  // Extend the sign bit and compare
  // If the input is matching (either 0 or all-ones) Xor result will be zero,
  // which is no-overflow.
  a->Eor(X(rtVixl), X(rtVixl), {X(i.d), vixl::ASR, 63});
  a->Cbz(X(rtVixl), &End);
  a->Mrs(X(rtVixl), NZCV);
  // Set the overflow bit
  a->Orr(X(rtVixl), X(rtVixl), {(int64_t)0x10000000});
  a->Msr(NZCV, X(rtVixl));
  a->bind(&End);
}

void Vgen::emit(const jcc &i) {
  assertx(i.cc != CC_None);
  if (i.targets[1] != i.targets[0]) {
    if (next == i.targets[1]) {
      // the taken branch is the fall-through block, invert the branch.
      return emit(jcc{ccNegate(i.cc), i.sf, {i.targets[1], i.targets[0]}});
    }
    // B.cond range is +/- 1MB but this uses BR
    auto start = emitSmashableJcc(*codeBlock, env.meta, kEndOfTargetChain, i.cc);
    jccs.push_back({start, i.targets[1]});
  }
  emit(jmp{i.targets[0]});
}

void Vgen::emit(const jcci& i) {
  emitSmashableJcc(*codeBlock, env.meta, i.taken, i.cc);
  emit(jmp{i.target});
}

void Vgen::emit(const jmp& i) {
  if (next == i.target) return;
  // B range is +/- 128MB but this uses BR
  auto start = emitSmashableJmp(*codeBlock, env.meta, kEndOfTargetChain);
  jmps.push_back({start, i.target});
}

void Vgen::emit(const jmpi& i) {
  auto const b = ssize_t(i.target);
  emit(ldimmqa{b, rtAsm});
  emit(jmpr{rtAsm, i.args});
}

void Vgen::emit(const jmpm& i) {
  prepAddr(i.target, rtAsm);
  a->Ldr(X(rtAsm), vixl::MemOperand(X(rtAsm)[0]));
  a->Br(X(rtAsm));
}

//Load an address as immediate. Required for relocation to work.
void Vgen::emit(const ldimmqa& i) {
  uint64_t ival = (uint64_t)i.s.q();
  emitSmashableMovq(*codeBlock, env.meta, ival, r64(Vreg64(size_t(i.d))));
}

void Vgen::emit(const lea& i) {
  prepAddr(i.s, i.d);
}

void Vgen::emit(const leap& i) {
  auto const b = ssize_t(i.s.r.disp);
  emit(ldimmqa{b, i.d});
}

void Vgen::emit(const loadb& i) {
  emit(loadzbl{i.s, rtAsm});
  emit(movb{Vreg8(size_t(rtAsm)), i.d});
}

void Vgen::emit(const loadqp& i) {
  auto const b = ssize_t(i.s.r.disp);
  emit(ldimmq{b, rtAsm});
  emit(load{rtAsm[0], i.d});
}

void Vgen::emit(const loadtqb& i) {
  emit(loadzbl{i.s, rtAsm});
  emit(movb{Vreg8(size_t(rtAsm)), i.d});
}

void Vgen::emit(const loadups& i) {
  // In arm, the scale should be equal to log2 of transfer size
  // As this is not true with the caller, we cannot use prepMem(m) here
  prepAddr(i.s, rtAsm);
  a->Ldr(Q(i.d), vixl::MemOperand(X(rtAsm)[0]));
}

void Vgen::emit(const loadzlq& i) {
  vixl::MemOperand m = prepMem(i.s);
  a->Ldr(W(Vreg32(size_t(i.d))), m);
  a->Uxtw(X(i.d), X(Vreg64(size_t(i.d))));
}

void Vgen::emit(const orwim& i) {
  // Cannot use rasm here as the conflict b/w prepMem(m) and store.
  // We can read it as signed as the upper bit is not used,
  // otherwise we may need to add a new instruction.
  emit(loadswq{i.m, rtVixl});
  emit(orqi{i.s0, rtVixl,  rtVixl, i.sf});
  emit(storew{Vreg16(size_t(rtVixl)), i.m});
}

///////////////////////////////////////////////////////////////////////////////

/* As the push & pop command in vixl expects sizes of 16 bytes*/
void Vgen::emit(const pop& i) {
  a->Mov(rAsm, rSp);
  a->Ldr(X(i.d), vixl::MemOperand(rAsm, 8, vixl::PostIndex));
  a->Mov(rSp, rAsm);
}

void Vgen::emit(const popm& i) {
  emit(pop{rtVixl});
  emit(store{rtVixl, i.d});
}

void Vgen::emit(const push& i) {
  a->Mov(rAsm, rSp);
  a->Str(X(i.s), vixl::MemOperand(rAsm, -8, vixl::PreIndex));
  a->Mov(rSp, rAsm);
}

///////////////////////////////////////////////////////////////////////////////

void Vgen::emit(const shlqi& i) {
  a->Lsl(X(i.d), X(i.s1), i.s0.l());
  // Set zero flag if result is zero
  a->Cmp(X(i.d), 0);
}

void Vgen::emit(const shrqi& i) {
  a->Mov(X(rtAsm), X(i.s1));
  a->Lsr(X(i.d), X(i.s1), i.s0.l());
  a->Tst(X(i.d), X(i.d));
  if(i.s0.l() != 0) {
    vixl::Label End;
    // Look at the last shifted out bit. There is a difference b/w
    // conditional flag Above for Arm and vasm representation. For vasm
    // CF=0 && ZF=0 is considered Above. For Arm it is CF=1 && ZF=0.
    a->Ubfx(X(rtAsm), X(rtAsm), i.s0.l()-1, 1);
    a->Cbnz(X(rtAsm), &End);
    a->Mrs(X(rtAsm), NZCV);
    // Set the carry bit
    a->Orr(X(rtAsm), X(rtAsm), {(int64_t)0x20000000});
    a->Msr(NZCV, X(rtAsm));
    a->bind(&End);
  }
}

void Vgen::emit(const srem& i) {
  // Passing dummy SF. It is not used anyway.
  emit(divsq {Vreg64(size_t(i.s0)), Vreg64(size_t(i.s1)), rtAsm});
  emit(imul {Vreg64(size_t(i.s1)), rtAsm, rtAsm, VregSF(size_t(0))});
  emit(subq {rtAsm, Vreg64(size_t(i.s0)),
       Vreg64(size_t(i.d)), VregSF(size_t(0))});
}

///////////////////////////////////////////////////////////////////////////////

/* Cannot use rasm here as the conflict b/w prep(m) and store */
void Vgen::emit(const storebi& i) {
  emit(ldimmb{i.s, rtVixl});
  emit(storeb{rtVixl, i.m});
}

void Vgen::emit(const storeli& i) {
  emit(ldimml{i.s, rtVixl});
  emit(storel{rtVixl, i.m});
}

void Vgen::emit(const storeqi& i) {
  auto const a = ssize_t(i.s.l());
  // ldimmq will sign extend the 32 bit of Immed.
  emit(ldimmq{a, rtVixl});
  emit(store{rtVixl, i.m});
}

///////////////////////////////////////////////////////////////////////////////

void Vgen::emit(const storeups& i) {
  // In arm, the scale should be equal to log2 of transfer size
  // As this is not true with the caller, we cannot use prepMem(m) here
  prepAddr(i.m, rtAsm);
  a->Str(Q(i.s), vixl::MemOperand(X(rtAsm)[0]));
}

void Vgen::emit(const subqi& i) {
  if(X(i.d).IsSP()) {
    a->Sub(X(i.d), X(i.s1), i.s0.l());
  } else {
    a->Sub(X(i.d), X(i.s1), i.s0.l(), SetFlags);
  }
}

void Vgen::emit(const tbcc& i) {
  assertx(i.cc == vixl::ne || i.cc == vixl::eq);
  if (i.targets[1] != i.targets[0]) {
    if (next == i.targets[1]) {
      // the taken branch is the fall-through block, invert the branch.
      return emit(tbcc{i.cc == vixl::ne ? vixl::eq : vixl::ne, i.bit, i.s,
               {i.targets[1], i.targets[0]}});
    }
    bccs.push_back({a->frontier(), i.targets[1]});
    // offset range +/- 32KB
    if (i.cc == vixl::ne) {
      a->tbnz(X(i.s), i.bit, 0);
    } else {
      a->tbz(X(i.s), i.bit, 0);
    }
  }
  emit(jmp{i.targets[0]});
}

void Vgen::emit(const testb& i) {
  // As test modifies the sign flag, the sign bit should
  // be taken care.
  emit(movsbl{i.s0, rtAsm});
  emit(movsbl{i.s1, rtVixl});
  emit(testl{rtAsm, rtVixl, i.sf});
}

void Vgen::emit(const testbim& i) {
  //As immediate are int32_t and it is converted to int64_t
  //in vixl, we need sign extension operation here.
  emit(loadsbq{i.s1, rtAsm});
  emit(testqib{i.s0, rtAsm, i.sf});
}

void Vgen::emit(const testlim& i) {
  //As immediate are int32_t and it is converted to int64_t
  //in vixl, we need sign extension operation here.
  emit(loadslq{i.s1, rtAsm});
  //As maximum immed is int32, we can use qi here.
  emit(testqi{i.s0, rtAsm, i.sf});
}

void Vgen::emit(const testqm& i) {
  emit(load{i.s1, rtAsm});
  emit(testq{i.s0, rtAsm, i.sf});
}

void Vgen::emit(const testqim& i) {
  emit(load{i.s1, rtAsm});
  emit(testqi{i.s0, rtAsm, i.sf});
}

void Vgen::emit(const testwim& i) {
  // As immediate are int32_t and it is converted to int64_t
  // in vixl, we need sign extension operation here.
  emit(loadswq{i.s1, rtAsm});
  emit(testqiw{i.s0, rtAsm, i.sf});
}

///////////////////////////////////////////////////////////////////////////////

/*
 * Some vasm opcodes don't have equivalent single instructions on ARM, and the
 * equivalent instruction sequences require scratch registers.  We have to
 * lower these to ARM-suitable vasm opcodes before register allocation.
 */
template<typename Lower>
void lower_impl(Vunit& unit, Vlabel b, size_t i, Lower lower) {
  auto& blocks = unit.blocks;
  auto const& vinstr = blocks[b].code[i];

  auto const scratch = unit.makeScratchBlock();
  SCOPE_EXIT { unit.freeScratchBlock(scratch); };
  Vout v(unit, scratch, vinstr.origin);

  lower(v);

  vector_splice(blocks[b].code, i, 1, blocks[scratch].code);
}

template<typename Inst>
void lower(Vunit& unit, Inst& inst, Vlabel b, size_t idx) {}

void lower(Vunit& unit, absdbl& i, Vlabel b, size_t idx) {
  lower_impl(unit, b, idx, [&] (Vout& v) {
  auto src = v.makeReg();
  auto dst = v.makeReg();
  v << copy{i.s, src};
  v << fabs{src, dst};
  v << copy{dst, i.d};
  });
}

void lower(Vunit& unit, cmpsd& i, Vlabel b, size_t idx) {
  lower_impl(unit, b, idx, [&] (Vout& v) {
  ConditionCode cond;
  auto true_val = v.makeReg();
  auto false_val = v.makeReg();
  auto result = v.makeReg();
  auto sf = v.makeReg();

  v << ldimmq{0xffffffffffffffff, true_val};
  v << ldimmq{0x0, false_val};

  if(i.pred == ComparisonPred::eq_ord)
    cond = CC_E;
  else if(i.pred == ComparisonPred::ne_unord)
    cond = CC_NE;
  else
    assertx(false);

  v << ucomisd{i.s1, i.s0, sf};
  v << cmovq{cond, sf, false_val, true_val, result};
  v << copy{result, i.d};
  });
}

void lower(Vunit& unit, cvttsd2siq& i, Vlabel b, size_t idx) {
  lower_impl(unit, b, idx, [&] (Vout& v) {
  auto fpsr = v.makeReg();
  auto res = v.makeReg();
  auto const sf = v.makeReg();
  auto err = v.makeReg();

  // Clear FPSR - TODO: Clear only IOC flag?
  v << setfpsr{v.cns(0)};

  // Do ARM64's double to signed int64 conversion.
  v << fcvtzs{i.s, res};

  // Error value
  v << ldimmq{0x8000000000000000, err};

  // Check if there was a conversion error
  v << cpfpsr{fpsr};
  v << testqib{1, fpsr, sf};

  // Move converted value or error
  v << cmovq{CC_NZ, sf, res, err, i.d};
  });
}

void lower(Vunit& unit, incqmlock& i, Vlabel b, size_t idx) {
  // Expecting only base register now.
  assertx(i.m.base.isValid() && !i.m.index.isValid() && (i.m.disp == 0));
  auto scratch = unit.makeReg();
  auto status = unit.makeReg();
  unit.blocks[b].code[idx] = incqexl{scratch, status, i.m, i.sf};
}


void lower(Vunit& unit, loadstubret& inst, Vlabel b, size_t i) {
  unit.blocks[b].code[i] = load{rsp()[8], inst.d};
}

void lower(Vunit& unit, stubtophp& inst, Vlabel b, size_t i) {
  unit.blocks[b].code[i] = lea{rsp()[16], rsp()};
}

///////////////////////////////////////////////////////////////////////////////

void lower_vcallarray(Vunit& unit, Vlabel b) {
  auto& code = unit.blocks[b].code;
  // vcallarray can only appear at the end of a block.
  auto const inst = code.back().get<vcallarray>();
  auto const origin = code.back().origin;

  auto argRegs = inst.args;
  auto const& srcs = unit.tuples[inst.extraArgs];
  jit::vector<Vreg> dsts;
  for (int i = 0; i < srcs.size(); ++i) {
    dsts.emplace_back(rarg(i));
    argRegs |= rarg(i);
  }
  /* Replace the existing instruction and add new ones */
  code.back() = copyargs{unit.makeTuple(srcs), unit.makeTuple(std::move(dsts))};
  code.emplace_back(callarray{inst.target, argRegs});
  code.back().origin = origin;
  code.emplace_back(unwind{{inst.targets[0], inst.targets[1]}});
  code.back().origin = origin;
}

void lowerArm(Vunit& unit, Vlabel b, size_t i) {
  Timer _t(Timer::vasm_lower);

  auto& inst = unit.blocks[b].code[i];

  switch (inst.op) {
#define O(name, ...)                    \
    case Vinstr::name:               \
    lower(unit, inst.name##_, b, i);  \
      break;

   VASM_OPCODES
#undef O
  }
}

void lowerArm(Vunit& unit) {
  // This pass relies on having no critical edges in the unit.
  splitCriticalEdges(unit);

  auto& blocks = unit.blocks;

  // The lowering operations for individual instructions may allocate scratch
  // blocks, which may invalidate iterators on `blocks'.  Correctness of this
  // pass relies on PostorderWalker /not/ using standard iterators on `blocks'.
  PostorderWalker{unit}.dfs([&] (Vlabel b) {
    assertx(!blocks[b].code.empty());
    for (size_t i = 0; i < blocks[b].code.size(); ++i) {
      lowerArm(unit, b, i);
    }
  });
}

/*
 * Lower a few abstractions to facilitate straightforward x64 codegen.
 */
void lowerForARM(Vunit& unit) {
  // This pass relies on having no critical edges in the unit.
  splitCriticalEdges(unit);

  // Scratch block can change blocks allocation, hence cannot use regular
  // iterators.
  auto& blocks = unit.blocks;

  PostorderWalker{unit}.dfs([&] (Vlabel ib) {
    assertx(!blocks[ib].code.empty());

    auto& back = blocks[ib].code.back();
    if (back.op == Vinstr::vcallarray) {
      lower_vcallarray(unit, Vlabel{ib});
    }

    for (size_t ii = 0; ii < blocks[ib].code.size(); ++ii) {
      vlower(unit, ib, ii);

      auto& inst = blocks[ib].code[ii];
      switch (inst.op) {
#define O(name, ...)                          \
        case Vinstr::name:                    \
          lower(unit, inst.name##_, ib, ii);  \
          break;

        VASM_OPCODES
#undef O
      }
    }
  });

  printUnit(kVasmLowerLevel, "after lower for Arm", unit);
}

///////////////////////////////////////////////////////////////////////////////
}

void finishARM(Vunit& unit, Vtext& text, CGMeta& fixups,
               const Abi& abi, AsmInfo* asmInfo) {
  optimizeExits(unit);
  if (!unit.constToReg.empty()) {
    foldImms<arm::ImmFolder>(unit);
  }
  lowerForARM(unit);
  simplify(unit);
  {
    Timer timer(Timer::vasm_copy);
    optimizeCopies(unit, abi);
  }
  if (unit.needsRegAlloc()) {
    Timer _t(Timer::vasm_xls);
    removeDeadCode(unit);
    allocateRegisters(unit, abi);
  }
  if (unit.blocks.size() > 1) {
    Timer _t(Timer::vasm_jumps);
    optimizeJmps(unit);
  }

  Timer _t(Timer::vasm_gen);
  vasm_emit<Vgen>(unit, text, fixups, asmInfo);
}

///////////////////////////////////////////////////////////////////////////////
}}
