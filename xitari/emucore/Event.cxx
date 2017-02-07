//============================================================================
//
//   SSSS    tt          lll  lll       
//  SS  SS   tt           ll   ll        
//  SS     tttttt  eeee   ll   ll   aaaa 
//   SSSS    tt   ee  ee  ll   ll      aa
//      SS   tt   eeeeee  ll   ll   aaaaa  --  "An Atari 2600 VCS Emulator"
//  SS  SS   tt   ee      ll   ll  aa  aa
//   SSSS     ttt  eeeee llll llll  aaaaa
//
// Copyright (c) 1995-2007 by Bradford W. Mott and the Stella team
//
// See the file "license" for information on usage and redistribution of
// this file, and for a DISCLAIMER OF ALL WARRANTIES.
//
// $Id: Event.cxx,v 1.11 2007/01/01 18:04:47 stephena Exp $
//============================================================================

#include "Event.hxx"
#include "EventStreamer.hxx"

using namespace ale;

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
Event::Event(EventStreamer* ev)
  : myNumberOfTypes(Event::LastType)
//ALE     myEventStreamer(ev)
{
  // Set all of the events to 0 / false to start with,
  // including analog paddle events.  Doing it this way
  // is a bit of a hack ...
  clear();

  myValues[PaddleZeroResistance]  =
  myValues[PaddleOneResistance]   =
  myValues[PaddleTwoResistance]   =
  myValues[PaddleThreeResistance] = 0;
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
Event::~Event()
{
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void Event::clear()
{
  for(int i = 0; i < myNumberOfTypes; ++i)
  {
    if(i != PaddleZeroResistance && i != PaddleOneResistance &&
       i != PaddleTwoResistance  && i != PaddleThreeResistance)
      myValues[i] = 0;
  }
}
