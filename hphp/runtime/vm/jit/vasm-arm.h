/*
   +----------------------------------------------------------------------+
   | HipHop for PHP                                                       |
   +----------------------------------------------------------------------+
   | Copyright (c) 2010-2015 Facebook, Inc. (http://www.facebook.com)     |
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

#ifndef incl_HPHP_JIT_VASM_ARM_H_
#define incl_HPHP_JIT_VASM_ARM_H_

#include "hphp/runtime/vm/jit/smashable-instr-arm.h"
#include "hphp/runtime/vm/jit/alignment.h"

namespace HPHP { namespace jit {

namespace arm {  

#define ARM_INSTR_WIDTH      4
#define VASM_ARM_CALL_SP_OFF 16

constexpr size_t callLrOff() { return ARM_INSTR_WIDTH; }

inline TCA addressSmashable(TCA addr)
{
  /* This will be using PC relative load for the label 
  *  But this relative offset remains same as we are not 
  *  modifying the contents of these set of instructions */
  vixl::Instruction *instr = vixl::Instruction::Cast(addr);
  auto label = reinterpret_cast<TCA>(instr->ImmPCOffsetTarget());
  long off = (long)(label - addr);
  TCA id;
  uint64_t mask = 0xffffffffffff0000ULL;

  /* Assuming size of 64 for smashable code */
  if(off >= 8 && off <= 64) {
    id = addr + off - 8;
    if(((*(uint64_t *)id) & mask) == (smashIdentifier(0) & mask))
      return id;
  }
  return NULL;
}

inline bool isInstrSmashableBranch(TCA addr, size_t& size)
{
  int off = 0;
  vixl::Instruction *instr = vixl::Instruction::Cast(addr);
  size = 4;
  /* Jcc starts with branch */
  if(instr->IsCondBranchImm()) {
    off = 4;
    instr = vixl::Instruction::Cast(addr + off);
  }
  if(instr->IsLoadLiteral()) {
    if(TCA id = addressSmashable(addr + off)) {
      if(((*(uint64_t *)id)  == (smashIdentifier((uint32_t)Alignment::SmashJmp))) ||
          (*(uint64_t *)id)  == (smashIdentifier((uint32_t)Alignment::SmashJcc))) {
        size = (id - addr) + 16;
        return true;
      }
    }
  }
  return false;
}

inline bool isInstrSmashableCall(TCA addr, size_t& size)
{
  vixl::Instruction *instr = vixl::Instruction::Cast(addr);
  size = 4;
  if(instr->IsLoadLiteral()) {
    if(TCA id = addressSmashable(addr)) {
      if((*(uint64_t *)id)  == (smashIdentifier((uint32_t)Alignment::SmashCall))) {
        size = (id - addr) + 16;
        return true;
      }
    }
  }
  return false;
}

inline bool isInstrSmashable(TCA addr, size_t& size)
{
  if(TCA id = addressSmashable(addr)) {
    size = (id - addr) + 16;
    return true;
  }  
  return false;
}

inline bool isCall(uint32_t op)
{
  return (((op & vixl::BLR) == vixl::BLR) ||
          ((op & vixl::BL) == vixl::BL));
}

inline int getCallLrOff(TCA addr)
{
  if(isCall(*(uint32_t *)(addr - (callLrOff() + ARM_INSTR_WIDTH)))) {
    return callLrOff();
  } else if(isCall(*(uint32_t *)(addr -
                   (smashableCallLrOff() + ARM_INSTR_WIDTH)))) {
    return smashableCallLrOff();
  }
  raise_error("Expecting call instruction in the address");
}

}//namespace arm

///////////////////////////////////////////////////////////////////////////////
#define VASM_AARCH64_LOWERING_INSTRS  \
struct loadpq {Vptr s; Vreg64 d1, d2; }; \
struct testqib { Immed s0; Vreg64 s1; VregSF sf; };\
struct testqiw { Immed s0; Vreg64 s1; VregSF sf; };\
struct cmplib { Immed  s0; Vreg32 s1; VregSF sf; };\
struct cmpqib { Immed  s0; Vreg64 s1; VregSF sf; };\
struct cmpqiw { Immed  s0; Vreg64 s1; VregSF sf; };\
struct loadsbq { Vptr s; Vreg64 d; };\
struct loadswq { Vptr s; Vreg64 d; };\
struct loadslq { Vptr s; Vreg64 d; };\
struct xorli { Immed s0; Vreg32 s1, d; VregSF sf; };\
/* Move sign extendes s to d */\
struct movsbl { Vreg8 s; Vreg32 d; }; \
struct loadexl { Vptr s; Vreg d; };\
struct storeexl { Vreg st, s ; Vptr d; };\
struct incqexl { Vreg s0,s1; Vptr m; VregSF sf; }; \
struct divsq { Vreg64 s0, s1, d; }; \
struct ldimmqa { Immed64 s; Vreg d; }; \
struct calldestr { Vptr target; RegSet args; }; \
struct cpfpsr { Vreg64 d; }; \
struct setfpsr { Vreg64 s; }; \
struct fcvtzs { VregDbl s; Vreg64 d;};

#define VASM_AARCH64_LOWERING_OPCODES \
  O(loadpq, Inone, U(s), D(d1) D(d2))\
  O(loadsbq, Inone, U(s), D(d))\
  O(loadswq, Inone, U(s), D(d))\
  O(loadslq, Inone, U(s), D(d))\
  O(testqib, I(s0), U(s1), D(sf))\
  O(testqiw, I(s0), U(s1), D(sf))\
  O(cmplib, I(s0), U(s1), D(sf))\
  O(cmpqib, I(s0), U(s1), D(sf))\
  O(cmpqiw, I(s0), U(s1), D(sf))\
  O(xorli, I(s0), UH(s1,d), DH(d,s1) D(sf))\
  O(movsbl, Inone, UH(s,d), DH(d,s))\
  O(loadexl, Inone, U(s), D(d))\
  O(storeexl, Inone, U(s) U(d), D(st))\
  O(incqexl, Inone, U(s0) U(s1) U(m), D(s0) D(s1) D(sf)) \
  O(divsq, Inone, U(s0) U(s1), D(d)) \
  O(ldimmqa, I(s), Un, D(d))\
  O(calldestr, Inone, U(target) U(args), Dn)\
  O(cpfpsr, Inone, Un, D(d))\
  O(setfpsr, Inone, U(s), Dn)\
  O(fcvtzs, Inone, U(s), D(d))\

#define 	VASM_AARCH64_LOWERING_INSTRS_EFFECT_FALSE \
    case Vinstr::loadpq:\
    case Vinstr::loadsbq:\
    case Vinstr::loadswq:\
    case Vinstr::loadslq:\
    case Vinstr::testqib:\
    case Vinstr::testqiw:\
    case Vinstr::cmplib:\
    case Vinstr::cmpqib:\
    case Vinstr::cmpqiw:\
    case Vinstr::xorli: \
    case Vinstr::movsbl:\
    case Vinstr::loadexl:\
    case Vinstr::incqexl:\
    case Vinstr::divsq: \
    case Vinstr::ldimmqa: \
    case Vinstr::cpfpsr: \
    case Vinstr::fcvtzs: \

#define 	VASM_AARCH64_LOWERING_INSTRS_EFFECT_TRUE \
    case Vinstr::storeexl: \
    case Vinstr::calldestr: \
    case Vinstr::setfpsr:

}}

#endif
