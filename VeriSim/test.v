module test;
    reg clk;
    reg rst;
    
    // Instantiate your design here
    
    // Clock generation
    initial begin
        $dumpfile("waveform.vcd");
        $dumpvars(0, test);
        clk = 0;
        forever #10 clk = ~clk;
    end
    
    initial begin
        rst = 1;
        #100 rst = 0;
        
        #1000 $finish;
    end
endmodule