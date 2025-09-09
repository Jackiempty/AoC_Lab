`include "src/PE_array/PE.sv"
`include "src/PE_array/GIN/GIN.sv"
`include "src/PE_array/GON/GON.sv"
`include "./include/define.svh"

module PE_array #(
    parameter NUMS_PE_ROW = `NUMS_PE_ROW,
    parameter NUMS_PE_COL = `NUMS_PE_COL,
    parameter XID_BITS = `XID_BITS,
    parameter YID_BITS = `YID_BITS,
    parameter DATA_SIZE = `DATA_BITS,
    parameter CONFIG_SIZE = `CONFIG_SIZE
)(
    input clk,
    input rst,

    // Scan Chain
    input set_XID,
    input [XID_BITS-1:0] ifmap_XID_scan_in,
    input [XID_BITS-1:0] filter_XID_scan_in,
    input [XID_BITS-1:0] ipsum_XID_scan_in,
    input [XID_BITS-1:0] opsum_XID_scan_in,

    input set_YID,
    input [YID_BITS-1:0] ifmap_YID_scan_in,
    input [YID_BITS-1:0] filter_YID_scan_in,
    input [YID_BITS-1:0] ipsum_YID_scan_in,
    input [YID_BITS-1:0] opsum_YID_scan_in,

    input set_LN,
    input [NUMS_PE_ROW-2:0] LN_config_in,

    // Controller
    input [NUMS_PE_ROW*NUMS_PE_COL-1:0] PE_en,
    input [CONFIG_SIZE-1:0] PE_config,
    input [XID_BITS-1:0] ifmap_tag_X,
    input [YID_BITS-1:0] ifmap_tag_Y,
    input [XID_BITS-1:0] filter_tag_X,
    input [YID_BITS-1:0] filter_tag_Y,
    input [XID_BITS-1:0] ipsum_tag_X,
    input [YID_BITS-1:0] ipsum_tag_Y,
    input [XID_BITS-1:0] opsum_tag_X,
    input [YID_BITS-1:0] opsum_tag_Y,

    // GLB
    input GLB_ifmap_valid,
    output logic GLB_ifmap_ready,
    input GLB_filter_valid,
    output logic GLB_filter_ready,
    input GLB_ipsum_valid,
    output logic GLB_ipsum_ready,
    input [DATA_SIZE-1:0] GLB_data_in,

    output logic GLB_opsum_valid,
    input GLB_opsum_ready,
    output logic [DATA_SIZE-1:0] GLB_data_out
);


    logic [NUMS_PE_ROW*NUMS_PE_COL-1:0] pe2noc_ready [0:2];   
    logic [NUMS_PE_ROW*NUMS_PE_COL-1:0] pe2noc_valid;     
    logic [DATA_SIZE*NUMS_PE_ROW*NUMS_PE_COL-1:0] pe2noc_data;
    logic [DATA_SIZE-1:0] pe_out_data [0:NUMS_PE_ROW*NUMS_PE_COL-1]; 


    logic [NUMS_PE_ROW*NUMS_PE_COL-1:0] noc2pe_valid [0:2]; 
    logic [NUMS_PE_ROW*NUMS_PE_COL-1:0] noc2pe_ready;
    logic [DATA_SIZE-1:0] noc2pe_data [0:2];


    logic [NUMS_PE_ROW*NUMS_PE_COL-1:0] ipsum_valid_arr;
    logic [DATA_SIZE-1:0] ipsum_data_arr [NUMS_PE_ROW*NUMS_PE_COL-1:0];


    logic [NUMS_PE_ROW-2:0] ln_setting;


    logic [NUMS_PE_ROW*NUMS_PE_COL-1:0] opsum_ready_arr;


    always_comb begin
        int row_idx;
        for (row_idx = 0; row_idx < NUMS_PE_ROW*NUMS_PE_COL; row_idx++)
            pe2noc_data[DATA_SIZE*row_idx +: DATA_SIZE] = pe_out_data[row_idx];
    end

    GIN gin_filter (
        .clk(clk),
        .rst(rst),
        .GIN_valid(GLB_filter_valid),
        .GIN_ready(GLB_filter_ready),
        .GIN_data(GLB_data_in),
        .tag_X(filter_tag_X),
        .tag_Y(filter_tag_Y),
        .set_XID(set_XID),
        .XID_scan_in(filter_XID_scan_in),
        .set_YID(set_YID),
        .YID_scan_in(filter_YID_scan_in),
        .PE_ready(pe2noc_ready[0]),
        .PE_valid(noc2pe_valid[0]),
        .PE_data(noc2pe_data[0])
    );

    GIN gin_ifmap (
        .clk(clk),
        .rst(rst),
        .GIN_valid(GLB_ifmap_valid),
        .GIN_ready(GLB_ifmap_ready),
        .GIN_data(GLB_data_in),
        .tag_X(ifmap_tag_X),
        .tag_Y(ifmap_tag_Y),
        .set_XID(set_XID),
        .XID_scan_in(ifmap_XID_scan_in),
        .set_YID(set_YID),
        .YID_scan_in(ifmap_YID_scan_in),
        .PE_ready(pe2noc_ready[1]),
        .PE_valid(noc2pe_valid[1]),
        .PE_data(noc2pe_data[1])
    );

    GIN gin_ipsum (
        .clk(clk),
        .rst(rst),
        .GIN_valid(GLB_ipsum_valid),
        .GIN_ready(GLB_ipsum_ready),
        .GIN_data(GLB_data_in),
        .tag_X(ipsum_tag_X),
        .tag_Y(ipsum_tag_Y),
        .set_XID(set_XID),
        .XID_scan_in(ipsum_XID_scan_in),
        .set_YID(set_YID),
        .YID_scan_in(ipsum_YID_scan_in),
        .PE_ready(pe2noc_ready[2]),
        .PE_valid(noc2pe_valid[2]),
        .PE_data(noc2pe_data[2])
    );

    GON gon_opsum (
        .clk(clk),
        .rst(rst),
        .GON_valid(GLB_opsum_valid),
        .GON_ready(GLB_opsum_ready),
        .GON_data(GLB_data_out),
        .tag_X(opsum_tag_X),
        .tag_Y(opsum_tag_Y),
        .set_XID(set_XID),
        .XID_scan_in(opsum_XID_scan_in),
        .set_YID(set_YID),
        .YID_scan_in(opsum_YID_scan_in),
        .PE_ready(noc2pe_ready),
        .PE_valid(pe2noc_valid),
        .PE_data(pe2noc_data)
    );


    always_ff @(posedge clk or posedge rst) begin
        if (rst)
            ln_setting <= '0;
        else if (set_LN)
            ln_setting <= LN_config_in;
        else
            ln_setting <= ln_setting;
    end
    

    genvar pe_idx;
    generate
        for (pe_idx = 0; pe_idx < NUMS_PE_ROW*NUMS_PE_COL; pe_idx++) begin: PE_GEN
            PE pe_inst (
                .clk(clk),
                .rst(rst),
                .PE_en(PE_en[pe_idx]),
                .i_config(PE_config),
                .ifmap(noc2pe_data[1]),
                .filter(noc2pe_data[0]),
                .ipsum(ipsum_data_arr[pe_idx]),
                .ifmap_valid(noc2pe_valid[1][pe_idx]),
                .filter_valid(noc2pe_valid[0][pe_idx]),
                .ipsum_valid(ipsum_valid_arr[pe_idx]),
                .opsum_ready(opsum_ready_arr[pe_idx]),
                .opsum(pe_out_data[pe_idx]),
                .ifmap_ready(pe2noc_ready[1][pe_idx]),
                .filter_ready(pe2noc_ready[0][pe_idx]),
                .ipsum_ready(pe2noc_ready[2][pe_idx]),
                .opsum_valid(pe2noc_valid[pe_idx])
            );
        end
    endgenerate

    always_comb begin
        int row_idx;
        for (row_idx = 0; row_idx < NUMS_PE_ROW*NUMS_PE_COL; row_idx++) begin
            if (row_idx >= NUMS_PE_ROW*NUMS_PE_COL - NUMS_PE_COL) begin
                ipsum_valid_arr[row_idx] = noc2pe_valid[2][row_idx];
                ipsum_data_arr[row_idx]  = noc2pe_data[2];
            end 
            else begin
                ipsum_valid_arr[row_idx] = (ln_setting[row_idx >> 3]) ? pe2noc_valid[row_idx + NUMS_PE_COL] : noc2pe_valid[2][row_idx];
                ipsum_data_arr[row_idx]  = (ln_setting[row_idx >> 3]) ? pe_out_data[row_idx + NUMS_PE_COL] : noc2pe_data[2];
            end
        end
    end

    always_comb begin
        int col_idx;
        for (col_idx = 0; col_idx < NUMS_PE_ROW*NUMS_PE_COL; col_idx = col_idx + 1) begin
            if (col_idx < NUMS_PE_COL) begin
                opsum_ready_arr[col_idx] = noc2pe_ready[col_idx];
            end 
            else begin
                opsum_ready_arr[col_idx] = (ln_setting[(col_idx >> 3) - 1]) ? pe2noc_ready[2][col_idx - NUMS_PE_COL] : noc2pe_ready[col_idx];
            end
        end
    end

endmodule
