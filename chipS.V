`timescale 1ns / 1ps

module neural_net(
    input wire clk,
    input wire rst,
    input wire signed [15:0] x_in,
    input wire signed [15:0] x_dot_in,
    input wire signed [15:0] theta_in,
    input wire signed [15:0] theta_dot_in,
    output reg signed [15:0] force_out
);

    // weights
    reg signed [15:0] w1 [31:0];
    reg signed [15:0] b1 [7:0];
    reg signed [15:0] w2 [7:0];
    reg signed [15:0] b2;

    initial begin
        // Layer 1 Weights (8 Neurons)
        w1[0] = -111; w1[1] = 112; w1[2] = -34; w1[3] = -39; 
        w1[4] = 19;   w1[5] = 34;  w1[6] = 141; w1[7] = 177; 
        w1[8] = -31;  w1[9] = 60;  w1[10] = 107; w1[11] = 178; 
        w1[12] = 5;   w1[13] = -27; w1[14] = 149; w1[15] = 103; 
        w1[16] = 8;   w1[17] = -15; w1[18] = -182; w1[19] = -81; 
        w1[20] = -15; w1[21] = -35; w1[22] = -332; w1[23] = -94; 
        w1[24] = -41; w1[25] = -19; w1[26] = 226; w1[27] = 79; 
        w1[28] = -86; w1[29] = -42; w1[30] = 28;  w1[31] = 11; 

        // Layer 1 Biases
        b1[0] = -34; b1[1] = 34; b1[2] = 71; b1[3] = 20; 
        b1[4] = 42;  b1[5] = 37; b1[6] = 86; b1[7] = 3; 

        // Layer 2 Weights (Linear Force)
        w2[0] = -6;  w2[1] = 76;  w2[2] = 51;  w2[3] = 36; 
        w2[4] = -80; w2[5] = -99; w2[6] = 28;  w2[7] = -4; 

        // Layer 2 Bias
        b2 = -7;
    end

    // SIGNALS
    reg signed [15:0] layer1_out [7:0]; 
    reg signed [31:0] accumulator;      
    integer i; 

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            force_out <= 0;
        end else begin
            
            // layer 1
            for (i = 0; i < 8; i = i + 1) begin
                // Input (Scale 2048) * Weight (Scale 128) = Scale 262,144
                // Bias is Scale 128. Wait...
                // CRITICAL: We trained with Bias Suppression (No Shift).
                // So we add Bias directly.
                accumulator = (x_in         * w1[i*4 + 0]) + 
                              (x_dot_in     * w1[i*4 + 1]) + 
                              (theta_in     * w1[i*4 + 2]) + 
                              (theta_dot_in * w1[i*4 + 3]) + 
                              (b1[i]); 
                
                // Scale Down: Shift 7 (Divide by 128)
                // Result is Scale 2048 (Q11).
                accumulator = accumulator >>> 7; 

                // ReLU 
                if (accumulator < 0) 
                    layer1_out[i] = 16'sd0;
                else if (accumulator > 32767) 
                    layer1_out[i] = 16'sd32767; //overflow management
                else 
                    layer1_out[i] = accumulator[15:0];
            end

            // layer 2
            // Linear Sum.
            accumulator = (b2 <<< 7); // Scale Bias to match multiplication
            for (i = 0; i < 8; i = i + 1) begin
                accumulator = accumulator + (layer1_out[i] * w2[i]);
            end
            
            // Normalize o/p
            accumulator = accumulator >>> 7;

            // Clamp to 16-bit Signed
            if (accumulator > 32767) force_out <= 32767;
            else if (accumulator < -32768) force_out <= -32768;
            else force_out <= accumulator[15:0];
        end
    end
endmodule