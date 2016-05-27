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
#include "hphp/runtime/vm/jit/relocation.h"

#include "hphp/runtime/vm/jit/align-arm.h"
#include "hphp/runtime/vm/jit/asm-info.h"
#include "hphp/runtime/vm/jit/cg-meta.h"
#include "hphp/runtime/vm/jit/ir-opcode.h"
#include "hphp/runtime/vm/jit/mc-generator.h"
#include "hphp/runtime/vm/jit/smashable-instr-arm.h"
#include "hphp/runtime/vm/jit/vasm-arm.h"
#include "hphp/vixl/a64/macro-assembler-a64.h"

namespace HPHP { namespace jit { namespace arm {

namespace {

TRACE_SET_MOD(hhir);

int relocateLiteralVal(TCA srcStart, size_t range, TCA srcCurrent, 
		TCA dstStart, TCA dstCurrent, CGMeta& meta)
{
  TCA addr = NULL, val;
  if(TCA id = addressSmashable(dstCurrent)) {
      val = reinterpret_cast<TCA>(*(uint64_t *)(id + 8));
      if((size_t)(val - srcStart) < range) {
        addr = dstStart + (val - srcStart);
      } else if (meta.addressImmediates.count((TCA)~uintptr_t(srcCurrent))) {
        // Handle weird, encoded offset, used by cgLdObjMethod intern mcprep
        always_assert(val == reinterpret_cast<TCA>((uintptr_t(srcCurrent) << 1) | 1));
        addr = reinterpret_cast<TCA>((uintptr_t(dstCurrent) << 1) | 1);
     }
     FTRACE(3, "Modify Address src {} dst {} oval {} nval {}\n",
			 srcCurrent, dstCurrent, 
			 (TCA) *(uint64_t *)(id + 8), addr);
     if(addr)
       *(uint64_t *)(id + 8) = (uint64_t)addr;
     return (int)((id - dstCurrent) + 16);
  } 
  return 0;
}

int relocateLiteralVal(RelocationInfo& rel, TCA addr)
{
  TCA val, adj;
  if(TCA id = addressSmashable(addr)) {
      val = reinterpret_cast<TCA>(*(uint64_t *)(id + 8));
      adj = rel.adjustedAddressAfter(val);
	  FTRACE(3, "Modify Address addr {} oval {} nval {}\n", 
			  addr, val, adj);
      if(adj)
        *(uint64_t *)(id + 8) = (uint64_t)adj;
      return (int)((id - addr) + 16);
  }
  return 0;
}


TCA getJmpAddress(TCA addr)
{
  vixl::Instruction *instr = vixl::Instruction::Cast(addr);
  if(instr->IsLoadLiteral()) {
    if(TCA id = addressSmashable(addr)) {
      if((*(uint64_t *)id)  == (smashIdentifier((uint32_t)Alignment::SmashJmp)))
       return(reinterpret_cast<TCA>(*(uint64_t *)(id + 8)));
    }
  } else if(instr->IsImmBranch()) {
    return reinterpret_cast<TCA>(instr->ImmPCOffsetTarget());
  }
  return NULL;  
}

bool isAccessPCRelative( vixl::Instruction *instr)
{
  return (instr->IsPCRelAddressing() || instr->IsLoadLiteral() || 
		  instr->IsImmBranch());
}

size_t relocateImpl(RelocationInfo& rel,
                    CodeBlock& destBlock,
                    TCA start, TCA end,
                    CGMeta& meta,
                    TCA* exitAddr) {
  TCA src = start;
  size_t range = end - src;
  TCA destStart = destBlock.frontier();
  TCA destStartPreReloc = destStart;
  int destOff = 0, instOff = 0, adjust = 0;
  TCA jmpDst = nullptr;

  FTRACE(3, "Relocate start {} end {} dstStart {} range {} \n", 
		  start, end, destStart, range);

  /* make sure source and dest are in same alignment. 
  *  Otherwise all the jmp/call smash alignment will go wrong */
  if(((long)start & 0x7) != ((long)destBlock.frontier() & 0x7)) {
    vixl::MacroAssembler a { destBlock };
    a.Nop();
    destStart += 4;
    adjust = 4;
    assertx((range + 4) <= destBlock.capacity());
  }
  destBlock.bytes(range, src);

  while (src != end) {
      assertx(src < end);
      jmpDst = NULL;
      instOff = 0;
      /* Don't need to align the instructions again, if we take care 
      *  of it above.  */
      /* All label based load/store/Adr/Br uses pc relative addressing */
      vixl::Instruction *instr = vixl::Instruction::Cast(src);
      if(isAccessPCRelative(instr)) {
        jmpDst = getJmpAddress(src);
        if(instr->IsLoadLiteral()) {
          instOff = relocateLiteralVal(start, range, src, 
                                       destStart, destStart + destOff, meta);
        }
        if(!instOff) {
          auto addr = reinterpret_cast<TCA>(instr->ImmPCOffsetTarget());
          // Try to modify the address only if it is out of this region,
          // may not succeed if the address range not in between +- 1MB
          // region(again size is dependent on the instruction type)
          // Otherwise no need to anything as the pc-relative offset
          // remains same.
          if((addr - start) >= range) {
             vixl::Instruction *dinst = vixl::Instruction::Cast(destStart + destOff);
             dinst->SetImmPCOffsetTarget(vixl::Instruction::Cast(addr));
          }
        }
      }
      instOff = instOff ? instOff : 4;
      if(src != start) {
        rel.recordAddress(src, destStart + destOff, 0);
        // fixup, catch block requires retAddress after the call
        if(instOff > 4) {
         size_t size;
         if(isInstrSmashableCall(src, size)) {
           rel.recordAddress(src + size - arm::smashableCallLrOff(), 
               destStart + destOff + size - arm::smashableCallLrOff(), 0);
         }
        }
      }
      src += instOff;
      destOff += instOff;
  } // while (src != end)

  if (exitAddr) {
    *exitAddr = jmpDst;
  }
  rel.recordRange(start, end, destStart-adjust, destBlock.frontier());
  rel.markAddressImmediates(meta.addressImmediates);

  // Flush cache for the destination text range
  __clear_cache((char *)destStartPreReloc, (char *)(destBlock.frontier()));

  // Return number of instructions processed
  return range/4 + adjust/4;
}

//////////////////////////////////////////////////////////////////////

}

/*
 * This should be called after calling relocate on all relevant ranges. It
 * will adjust all references into the original src ranges to point into the
 * corresponding relocated ranges.
 */
void adjustForRelocation(RelocationInfo& rel) {
  for (const auto& range : rel.srcRanges()) {
    arm::adjustForRelocation(rel, range.first, range.second);
  }
}

/*
 * This will update a single range that was not relocated, but that
 * might refer to relocated code (such as the cold code corresponding
 * to a tracelet). Unless its guaranteed to be all position independent,
 * its "meta" should have been passed into a relocate call earlier.
 */
void adjustForRelocation(RelocationInfo& rel, TCA srcStart, TCA srcEnd) {
  auto start = rel.adjustedAddressAfter(srcStart);
  auto end = rel.adjustedAddressBefore(srcEnd);
  int instOff;
  FTRACE(3, "Adjust srcStart {} srcEnd {} start {} end {}\n", 
		  srcStart, srcEnd, start, end);
  if (!start) {
    start = srcStart;
    end = srcEnd;
  } else {
    always_assert(end);
  }
  while (start != end) {
    assertx(start < end);
    instOff = 0;
    vixl::Instruction *instr = vixl::Instruction::Cast(start);
    if(isAccessPCRelative(instr)) {
      if(instr->IsLoadLiteral()) {
        instOff = relocateLiteralVal(rel, start);
      }
      if(!instOff) {
        auto addr = reinterpret_cast<TCA>(instr->ImmPCOffsetTarget());
        /*
        * A pointer into something that has been relocated needs to be
        * updated.
        */
        if (TCA adjusted = rel.adjustedAddressAfter(addr)) {
          instr->SetImmPCOffsetTarget(vixl::Instruction::Cast(adjusted));
        }
      }
    }
    start += instOff ? instOff : 4;
  }
}

/*
 * Adjust potentially live references that point into the relocated
 * area.
 * Must not be called until its safe to run the relocated code.
 */
void adjustCodeForRelocation(RelocationInfo& rel, CGMeta& meta) {
  int instOff;
  for (auto addr : meta.reusedStubs) {
    /*
    * The stubs are terminated by a ud2. Check for it.
    */
    FTRACE(3, "Adjust addr {} \n", addr);
    while (((*(uint32_t *)addr) & 0xffff0000) != BRK_INSTR(0)) {
      instOff = 0;
      vixl::Instruction *instr = vixl::Instruction::Cast(addr);
      if(isAccessPCRelative(instr)) {
        if(instr->IsLoadLiteral()) {
          instOff = relocateLiteralVal(rel, addr);
        }
        if(!instOff) {
           auto adr = reinterpret_cast<TCA>(instr->ImmPCOffsetTarget());
           /*
           * A pointer into something that has been relocated needs to be
           * updated.
           */
           if (TCA adjusted = rel.adjustedAddressAfter(adr)) {
             instr->SetImmPCOffsetTarget(vixl::Instruction::Cast(adjusted));
           }
         }
       }
       addr += instOff ? instOff : 4;
     }
  }

  for (auto codePtr : meta.codePointers) {
    if (TCA adjusted = rel.adjustedAddressAfter(*codePtr)) {
      *codePtr = adjusted;
    }
  }
}

void findFixups(TCA start, TCA end, CGMeta& meta) {
  size_t size;
  TCA retAddr;
  FTRACE(3, "FindFixup start {} end {}\n",start, end);
  while (start != end) {
    assert(start < end);
    retAddr = NULL;
    if(arm::isInstrSmashable(start, size)) {
      if(arm::isInstrSmashableCall(start, size)) 
       retAddr = start + size - arm::smashableCallLrOff();
    } else {
      if(arm::isCall(*(uint32_t *)start))
        retAddr = start + 4;
        size = 4;
    }

    if (retAddr) {
      if (auto fixup = mcg->fixupMap().findFixup(retAddr)) {
        meta.fixups.emplace_back(retAddr, *fixup);
      }
      if (auto ct = mcg->catchTraceMap().find(retAddr)) {
        meta.catches.emplace_back(retAddr, *ct);
      }
    }
    start += size;
  }
}


/*
 * Relocate code in the range start, end into dest, and record
 * information about what was done to rel.
 * On exit, internal references (references into the source range)
 * will have been adjusted (ie they are still references into the
 * relocated code). External code references continue to point to
 * the same address as before relocation.
 */
size_t relocate(RelocationInfo& rel,
                CodeBlock& destBlock,
                TCA start, TCA end,
                CGMeta& meta,
                TCA* exitAddr) {
   return relocateImpl(rel, destBlock, start, end, meta, exitAddr);
}

//////////////////////////////////////////////////////////////////////

}}}
