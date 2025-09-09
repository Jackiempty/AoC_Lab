`include "./include/define.svh"

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
    output logic signed [`DATA_BITS-1:0] opsum,
    output logic ifmap_ready,
    output logic filter_ready,
    output logic ipsum_ready,
    output logic opsum_valid
);

    typedef enum logic [2:0] {
        S_IDLE,
        S_LOAD_FILTER,
        S_LOAD_IFMAP,
        S_LOAD_IPSUM,
        S_CALC,
        S_WRITE_OPSUM,
        S_FINISH
    } pe_state_t;

    pe_state_t state, state_nxt;


    logic cfg_mode;
    logic [1:0] cfg_o_ch, cfg_i_ch;
    logic [4:0] cfg_o_col;

    logic [5:0] filt_idx;


    logic signed [`FILTER_SIZE-1:0] filt_buf [0:`FILTER_SPAD_LEN-1];
    logic signed [`IFMAP_SIZE-1:0] fmap_buf [0:`IFMAP_SPAD_LEN-1];
    logic signed [`PSUM_SIZE-1:0] psum_buf [0:`OFMAP_SPAD_LEN-1];
    logic signed [`PSUM_SIZE-1:0] mult_buf [0:`OFMAP_SPAD_LEN-1];


    logic [3:0] cnt_main;
    logic [5:0] cnt_mult;
    logic [3:0] cnt_fmap;
    logic [2:0] cnt_out;
    logic [3:0] cnt_mult_fmap;
    logic [`OFMAP_INDEX_BIT-1:0] cnt_mult_idx;
    logic [`OFMAP_COL_BIT-1:0] cnt_col;

    integer idx;


    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            cfg_mode  <= '0;
            cfg_o_ch  <= '0;
            cfg_o_col <= '0;
            cfg_i_ch  <= '0;
        end 
        else if (PE_en) begin
            cfg_mode  <= i_config[`CONFIG_SIZE-1];
            cfg_o_ch  <= i_config[`CONFIG_SIZE-2:`CONFIG_SIZE-3];
            cfg_o_col <= i_config[`CONFIG_SIZE-4:`CONFIG_SIZE-8];
            cfg_i_ch  <= i_config[`CONFIG_SIZE-9:0];
        end
        else begin
            cfg_mode  <= cfg_mode;
            cfg_o_ch  <= cfg_o_ch;
            cfg_o_col <= cfg_o_col;
            cfg_i_ch  <= cfg_i_ch;
        end
    end


    always_ff @(posedge clk or posedge rst) begin
        if (rst) cnt_main <= '0;
        else if (state != state_nxt) cnt_main <= '0;
        else 
            case (state)
                S_LOAD_FILTER: cnt_main <= filter_valid ? cnt_main + 1 : cnt_main;
                S_LOAD_IPSUM: cnt_main <= ipsum_valid ? cnt_main + 1 : cnt_main;
                S_WRITE_OPSUM: cnt_main <= opsum_ready ? cnt_main + 1 : cnt_main;
                default: cnt_main <= cnt_main;
            endcase
    end


    always_comb filt_idx = cnt_main * cfg_i_ch + {2'd0, cnt_main};

    always_ff @(posedge clk or posedge rst) begin
        if (rst) 
            cnt_fmap <= '0;
        else if (state == S_LOAD_IFMAP && ifmap_valid) 
            cnt_fmap <= cnt_fmap + {2'd0, cfg_i_ch} + 4'd1;
        else if (state == S_WRITE_OPSUM && state_nxt == S_LOAD_IFMAP)
            cnt_fmap <= cnt_fmap - {2'd0, cfg_i_ch} - 4'd1;
        else cnt_fmap <= cnt_fmap;
    end

    always_ff @(posedge clk or posedge rst) begin
        if (rst) cnt_mult_fmap <= '0;
        else if (state == S_CALC)
            cnt_mult_fmap <= 
                (cnt_mult_fmap == (`FILT_R * cfg_i_ch + `FILT_R - 1)) ? 4'd0 : cnt_mult_fmap + 1;
        else cnt_mult_fmap <= '0;
    end

    always_ff @(posedge clk or posedge rst) begin
        if (rst) 
            cnt_mult <= '0;
        else
            cnt_mult <= state == S_CALC ? cnt_mult + 6'd1 : '0;
    end


    always_ff @(posedge clk or posedge rst) begin
        if (rst)
            cnt_mult_idx <= '0;
        else if (state == S_CALC)
            if (cnt_mult_fmap == (`FILT_R * cfg_i_ch + `FILT_R - 1))
                cnt_mult_idx <= cnt_mult_idx + `OFMAP_INDEX_BIT'd1;
            else cnt_mult_idx <= cnt_mult_idx;
        else begin
            cnt_mult_idx <= '0;
        end
    end

    always_ff @(posedge clk or posedge rst) begin
        if (rst) cnt_col <= '0;
        else if (state == S_WRITE_OPSUM && cnt_out == {1'b0, cfg_o_ch} + 3'd1 && opsum_ready)
            cnt_col <= cnt_col + 1;
        else cnt_col <= cnt_col;
    end

        
    always_ff @(posedge clk or posedge rst) begin
        if (rst) cnt_out <= '0;
        else if (state == S_LOAD_IFMAP) cnt_out <= 3'd0;
        else if (state == S_LOAD_IPSUM && state_nxt == S_WRITE_OPSUM) cnt_out <= cnt_out + 1;
        else if (opsum_ready && opsum_valid) cnt_out <= cnt_out + 1;
        else cnt_out <= cnt_out;
    end

    always_ff @(posedge clk or posedge rst) begin
        if (rst) filt_buf <= '{default: '0};
        else if (state == S_LOAD_FILTER && filter_valid) begin
            filt_buf <= filt_buf;
            for (idx = 0; idx < 4; idx++) begin
                if (idx[1:0] <= cfg_i_ch )
                    filt_buf[{26'd0,filt_idx} + idx] <= filter[(8*idx)+:8];
            end
        end
        else filt_buf <= filt_buf;
    end
    
    always_comb filter_ready = (state == S_LOAD_FILTER);

    always_ff @(posedge clk or posedge rst) begin
        if (rst) 
            fmap_buf <= '{default: '0};
        else if (state == S_LOAD_IFMAP && ifmap_valid) begin
            fmap_buf <= fmap_buf;
            for (idx = 0; idx < 4; idx++) begin
                if (idx[1:0] <= cfg_i_ch)
                    fmap_buf[{28'd0,cnt_fmap} + idx] <= $signed(ifmap[(8*idx)+:8] ^ `IFMAP_SIZE'd128);
            end
        end 
        else if (state == S_WRITE_OPSUM && state_nxt == S_LOAD_IFMAP) begin
            fmap_buf <= fmap_buf;
            for (idx = 0; idx < `IFMAP_SPAD_LEN-{30'd0, cfg_i_ch}-1; idx++)
                fmap_buf[idx] <= fmap_buf[idx+{30'd0, cfg_i_ch}+1];
        end
        else fmap_buf <= fmap_buf;
    end

    always_comb ifmap_ready = (state == S_LOAD_IFMAP);

    always_ff @(posedge clk or posedge rst) begin
        if (rst) mult_buf <= '{default: '0};
        else if (state == S_CALC) begin
            mult_buf[cnt_mult_idx] <= mult_buf[cnt_mult_idx] + filt_buf[cnt_mult] * fmap_buf[cnt_mult_fmap];
        end 
        else if (state == S_LOAD_IFMAP) begin
            mult_buf <= '{default: '0};
        end
        else mult_buf <= mult_buf;
    end

    always_ff @(posedge clk or posedge rst) begin
        if (rst) psum_buf <= '{default: '0};
        else if (state == S_LOAD_IPSUM && ipsum_valid)
            psum_buf[cnt_main[1:0]] <= mult_buf[cnt_main[1:0]] + ipsum;
        else psum_buf <= psum_buf;
    end

    always_comb ipsum_ready = (state == S_LOAD_IPSUM);

    always_ff @(posedge clk or posedge rst) begin
        if (rst) opsum <= '0;
        else if (state == S_LOAD_IPSUM && state_nxt == S_WRITE_OPSUM) opsum <= psum_buf[0];
        else if (state == S_WRITE_OPSUM && opsum_ready && opsum_valid) opsum <= psum_buf[cnt_out[1:0]];
        else opsum <= opsum;
    end

    always_comb opsum_valid = (state == S_WRITE_OPSUM);

    always_comb begin
        case (state)
            S_IDLE:         state_nxt = S_LOAD_FILTER;
            S_LOAD_FILTER:  state_nxt = pe_state_t'((cnt_main == `FILT_R * cfg_o_ch + `FILT_R - 1 && filter_valid) ? S_LOAD_IFMAP : S_LOAD_FILTER);
            S_LOAD_IFMAP:   state_nxt = pe_state_t'((cnt_fmap == `FILT_R * cfg_i_ch + `FILT_R - {2'd0, cfg_i_ch} - 4'd1 && ifmap_valid) ? S_CALC : S_LOAD_IFMAP);
            S_CALC:         state_nxt = pe_state_t'((cnt_mult == (`FILT_R * cfg_i_ch * cfg_o_ch + cfg_i_ch * `FILT_R + cfg_o_ch * `FILT_R + `FILT_R - 1)) ? S_LOAD_IPSUM : S_CALC);
            S_LOAD_IPSUM:   state_nxt = pe_state_t'((cnt_main[1:0] == cfg_o_ch && ipsum_valid) ? S_WRITE_OPSUM : S_LOAD_IPSUM);
            S_WRITE_OPSUM:  state_nxt = pe_state_t'((cnt_out == {1'b0, cfg_o_ch} + 3'd1 && opsum_ready) ?
                                        ((cnt_col == cfg_o_col) ? S_FINISH : S_LOAD_IFMAP) : S_WRITE_OPSUM);
            S_FINISH:       state_nxt = S_FINISH;
            default:        state_nxt = S_IDLE;
        endcase
    end

    always_ff @(posedge clk or posedge rst) begin
        if (rst) state <= S_IDLE;
        else state <= state_nxt;
    end

endmodule
