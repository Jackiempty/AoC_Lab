`include "define.svh"
module PE (
    input clk,
    input rst,
    input PE_en,
    input [`CONFIG_SIZE-1:0] i_config,
    input [`DATA_BITS-1:0] ifmap,
    input [`DATA_BITS-1:0] filter,
    input [`DATA_BITS-1:0] ipsum,
    input ifmap_valid,
    input filter_valid,
    input ipsum_valid,
    input opsum_ready,
    output logic [`DATA_BITS-1:0] opsum,
    output logic ifmap_ready,
    output logic filter_ready,
    output logic ipsum_ready,
    output logic opsum_valid
);

// --- Config decode ---
logic        is_fc;
logic [1:0]  p;           // not used in this primitive
logic [4:0]  F;
logic [1:0]  q;
logic [6:0]  total_ops;   // F × q, max 32 × 4 = 128

always_comb begin
    is_fc     = i_config[9];
    p         = i_config[8:7] + 1;
    F         = i_config[6:2] + 1;
    q         = i_config[1:0] + 1;
    total_ops = F * q;
end

// --- FSM states ---
typedef enum logic [1:0] {
    IDLE,
    COMPUTE,
    WAIT_OUT
} state_t;

state_t state, next_state;

// --- Counters ---
logic [6:0] op_cnt;

// --- Internal signals ---
logic signed [7:0]  ifmap_s;
logic signed [31:0] filter_s;
logic signed [31:0] ipsum_s;
logic signed [31:0] mac_result;
logic signed [31:0] accum;

// --- Zero-point subtraction ---
always_comb begin
    ifmap_s  = $signed(ifmap[7:0]) - 8'sd128;
    filter_s = $signed(filter);
    ipsum_s  = $signed(ipsum);
end

assign mac_result = ifmap_s * filter_s;

// --- FSM state transitions ---
always_ff @(posedge clk or posedge rst) begin
    if (rst) begin
        state <= IDLE;
    end else begin
        state <= next_state;
    end
end

always_comb begin
    next_state = state;
    case (state)
        IDLE: begin
            if (PE_en) next_state = COMPUTE;
        end
        COMPUTE: begin
            if (ifmap_valid && filter_valid && ipsum_valid) begin
                if (op_cnt == total_ops - 1)
                    next_state = WAIT_OUT;
            end
        end
        WAIT_OUT: begin
            if (opsum_ready) next_state = IDLE;
        end
    endcase
end

// --- MAC computation + accum ---
always_ff @(posedge clk or posedge rst) begin
    if (rst) begin
        accum      <= 0;
        op_cnt     <= 0;
        opsum_valid <= 0;
    end else begin
        case (state)
            IDLE: begin
                accum      <= 0;
                op_cnt     <= 0;
                opsum_valid <= 0;
            end

            COMPUTE: begin
                if (ifmap_valid && filter_valid && ipsum_valid) begin
                    accum  <= ipsum_s + mac_result;
                    op_cnt <= op_cnt + 1;
                end
            end

            WAIT_OUT: begin
                opsum_valid <= 1;
                if (opsum_ready)
                    opsum_valid <= 0;
            end
        endcase
    end
end

// --- Output assignments ---
assign opsum        = accum;
assign ifmap_ready  = (state == COMPUTE);
assign filter_ready = (state == COMPUTE);
assign ipsum_ready  = (state == COMPUTE);

endmodule
