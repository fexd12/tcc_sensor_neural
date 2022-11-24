#include "Brain.h"
#include "stdio.h"
#include <iostream>
#include <string> // header file for string


Brain::Brain()
{
    // Keep the rest of the initialization process in a separate method in case
    // we overload the constructor.
    init();
}

void Brain::init()
{
    // It's up to the calling code to start the stream
    freshPacket = false;
    inPacket = false;
    packetIndex = 0;
    packetLength = 0;
    eegPowerLength = 0;
    hasPower = false;
    checksum = 0;
    checksumAccumulator = 0;

    signalQuality = 200;
    attention = 0;
    meditation = 0;

    clearEegPower();
}

bool Brain::update(uint8_t latestByte)
{
    // Build a packet if we know we're and not just listening for sync bytes.
    if (inPacket)
    {

        // First byte after the sync bytes is the length of the upcoming packet.
        if (packetIndex == 0)
        {
            packetLength = latestByte;

            // Catch error if packet is too long
            if (packetLength > MAX_PACKET_LENGTH)
            {
                // Packet exceeded max length
                // Send an error
                // printf("ERROR: Packet too long %i", packetLength);
                inPacket = false;
            }
        }
        else if (packetIndex <= packetLength)
        {
            // Run of the mill data bytes.

            // Print them here

            // Store the byte in an array for parsing later.
            packetData[packetIndex - 1] = latestByte;

            // Keep building the checksum.
            checksumAccumulator += latestByte;
        }
        else if (packetIndex > packetLength)
        {
            // We're at the end of the data payload.

            // Check the checksum.
            checksum = latestByte;
            // checksumAccumulator = 255 - checksumAccumulator;

            checksumAccumulator &= 0xFF;
            checksumAccumulator = ~checksumAccumulator & 0xFF;

            // Do they match?
            if (checksum == checksumAccumulator)
            {
                bool parseSuccess = parsePacket();

                if (parseSuccess)
                {
                    freshPacket = true;
                }
                else
                {
                    // Parsing failed, send an error.
                    // printf("ERROR: Could not parse");
                    // good place to print the packet if debugging
                }
            }
            else
            {
                // Checksum mismatch, send an error.
                // printf("ERROR: Checksum");
                // good place to print the packet if debugging
            }
            // End of packet

            // Reset, prep for next packet
            inPacket = false;
        }

        packetIndex++;
    }

    // Look for the start of the packet
    if ((latestByte == 170) && (lastByte == 170) && !inPacket)
    {
        // Start of packet
        inPacket = true;
        packetIndex = 0;
        checksumAccumulator = 0;
    }

    // Keep track of the last byte so we can find the sync byte pairs.
    lastByte = latestByte;

    if (freshPacket)
    {
        freshPacket = false;
        return true;
    }
    else
    {
        return false;
    }
}

void Brain::clearPacket()
{
    for (uint8_t i = 0; i < MAX_PACKET_LENGTH; i++)
    {
        packetData[i] = 0;
    }
}

void Brain::clearEegPower()
{
    // Zero the power bands.
    for (uint8_t i = 0; i < EEG_POWER_BANDS; i++)
    {
        eegPower[i] = 0;
    }
}

bool Brain::parsePacket()
{
    // Loop through the packet, extracting data.
    // Based on mindset_communications_protocol.pdf from the Neurosky Mindset SDK.
    // Returns true if passing succeeds
    hasPower = false;
    bool parseSuccess = true;
    // int rawValue = 0;

    clearEegPower(); // clear the eeg power to make sure we're honest about missing values

    for (uint8_t i = 0; i < packetLength; i++)
    {
        switch (packetData[i])
        {
        case 0x2:
            signalQuality = packetData[++i];
            break;
        case 0x4:
            attention = packetData[++i];
            break;
        case 0x5:
            meditation = packetData[++i];
            break;
        case 0x83:
            // ASIC_EEG_POWER: eight big-endian 3-uint8_t unsigned integer values representing delta, theta, low-alpha high-alpha, low-beta, high-beta, low-gamma, and mid-gamma EEG band power values
            // The next uint8_t sets the length, usually 24 (Eight 24-bit numbers... big endian?)
            // We dont' use this value so let's skip it and just increment i
            i++;

            // Extract the values
            for (int j = 0; j < EEG_POWER_BANDS; j++)
            {
                uint8_t a, b, c;
                a = packetData[++i];
                b = packetData[++i];
                c = packetData[++i];
                eegPower[j] = ((uint32_t)a << 16) | ((uint32_t)b << 8) | (uint32_t)c;
            }

            hasPower = true;
            // This seems to happen once during start-up on the force trainer. Strange. Wise to wait a couple of packets before
            // you start reading.
            break;
        case 0x80:
            // We dont' use this value so let's skip it and just increment i
            //uint8_t packetLength = packetData[++i];
            //rawValue = ((int)packetData[++i] << 8) | packetData[++i];
            i += 3;
            break;
        default:
            // Broken packet ?
            /*
            Serial.print(F("parsePacket UNMATCHED data 0x"));
            Serial.print(packetData[i], HEX);
            Serial.print(F(" in position "));
            Serial.print(i, DEC);
            printPacket();
            */
            parseSuccess = false;
            break;
        }
    }
    return parseSuccess;
}

char *Brain::readErrors()
{
    return latestError;
}

uint32_t *Brain::readCSV()
{
    // spit out a big string?
    // find out how big this really needs to be
    // should be popped off the stack once it goes out of scope?
    // make the character array as small as possible
    // char csvBuffer_temp[100] = {};

    // char *tmp = csvBuffer_temp;

    if (hasPower)
    {
        // int size;
        
        // size =  
        //     sizeof(eegPower[0]) +
        //     sizeof(eegPower[1]) +
        //     sizeof(eegPower[2]) +
        //     sizeof(eegPower[3]) +
        //     sizeof(eegPower[4]) +
        //     sizeof(eegPower[5]) +
        //     sizeof(eegPower[6]) +
        //     sizeof(eegPower[7]) + 
        //     1;
        return eegPower;

        // snprintf(csvBuffer_temp, sizeof(csvBuffer_temp),"%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu",
        //         eegPower[0],
        //         eegPower[1],
        //         eegPower[2],
        //         eegPower[3],
        //         eegPower[4],
        //         eegPower[5],
        //         eegPower[6],
        //         eegPower[7]);
        // return tmp;
    }
    return 0;
}

uint8_t Brain::readSignalQuality()
{
    return signalQuality;
}

uint8_t Brain::readAttention()
{
    return attention;
}

uint8_t Brain::readMeditation()
{
    return meditation;
}

uint32_t *Brain::readPowerArray()
{
    return eegPower;
}

uint32_t Brain::readDelta()
{
    return eegPower[0];
}

uint32_t Brain::readTheta()
{
    return eegPower[1];
}

uint32_t Brain::readLowAlpha()
{
    return eegPower[2];
}

uint32_t Brain::readHighAlpha()
{
    return eegPower[3];
}

uint32_t Brain::readLowBeta()
{
    return eegPower[4];
}

uint32_t Brain::readHighBeta()
{
    return eegPower[5];
}

uint32_t Brain::readLowGamma()
{
    return eegPower[6];
}

uint32_t Brain::readMidGamma()
{
    return eegPower[7];
}
