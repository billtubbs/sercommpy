# LED Display Commands

| Code | Byte Data                                         |  Description                                   |
| ---- | ------------------------------------------------- | ---------------------------------------------- |
| L1   | [I, J, R, G, B]                                   | Set the colour of one LED                      |
| LN   | [N1, N2, I1, J1, R1, G1, B1, ..., IN, JN, RN, GN, BN]  | Set the colours of N LEDs                       |
| LA   | [R1, G1, B1, ..., RL, GL, BL]                     | Set all LED colours                            |
| CN   | [N1, N2, R, G, B, I1, J1, ..., IN, JN]            | Set N LEDs to one colour                       |
| CA   | [R, G, B]                                         | Set all LEDs to one colour                     |
| LB   | [B]                                               | Set brightness level (affects all LEDs)        |
| LC   |                                                   | Clear screen (to black)                        |
| G1   | [I, J]                                            | Get the colour of an LED                       |
| GB   |                                                   | Get the brightness reading from photoresistor  |
| GT   |                                                   | Get the current clock time                     |
| SN   |                                                   | Show LED updates now                           |
| SA   | [T1, T2]                                          | Show LED updates at clock time                 |
| RR   |                                                   | Report when ready to show updates              |


Key to symbols

 - I, J : High and low bytes of the integer ID of an LED (0-65536)
 - R, G, B : Red, green, and blue color intensities
 - N1, N2 : High and low bytes of unsigned long integer value (0-65536)
 - B : Brightness level (0-5) (TODO: Confirm this)
 - T1, T2 : High and low bytes of the clock time in milliseconds (0-65536)