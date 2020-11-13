use RahavardNovin3;

if object_id('dbo.jstring', 'FN') is not null drop function dbo.jstring;
if object_id('dbo.jprint', 'FN') is not null drop function dbo.jprint;
if object_id('dbo.jyear', 'FN') is not null drop function dbo.jyear;
if object_id('dbo.jmonth', 'FN') is not null drop function dbo.jmonth;
if object_id('dbo.jday', 'FN') is not null drop function dbo.jday;
if object_id('dbo.jalali', 'FN') is not null drop function dbo.jalali;
if object_id('dbo.jalaliw', 'FN') is not null drop function dbo.jalaliw;
if object_id('dbo.jparse', 'FN') is not null drop function dbo.jparse;

if object_id('dbo.jmax', 'FN') is not null drop function dbo.jmax;
if object_id('dbo.jleap', 'FN') is not null drop function dbo.jleap;
go

create function dbo.jleap (@year int) returns bit with encryption
begin
	declare @leapDays0 float, @leapDays1 float

	if @year > 0 begin
		-- 0.24219 ~  extra days of  one year
		set @leapDays0 = ((@year + 38) % 2820) * 0.24219 + 0.025; 
        -- 38 days is the difference of epoch to 2820-year/ cycle
		set @leapDays1 = ((@year + 39) % 2820) * 0.24219 + 0.025; 
	end else if @year < 0 begin
		set @leapDays0 = ((@year + 39) % 2820) * 0.24219 + 0.025;
		set @leapDays1 = ((@year + 40) % 2820) * 0.24219 + 0.025;
	end else return 0
	
	declare @frac0 int, @frac1 int

	set @frac0 = floor((@leapDays0 - floor(@leapDays0)) * 1000);
	set @frac1 = floor((@leapDays1 - floor(@leapDays1)) * 1000);

	return case when @frac0 <= 266 and @frac1 > 266 then 1 else 0 end;
end
go

create function dbo.jmax (@year int, @month int) returns int with encryption
begin
	if @month <= 0 or @month > 12 begin
		-- raiserror ('Jalali Max (value:month) month must be between 1 and 12.', 18, 1, 'dbo.jmax')
		return null
	end else if @month <= 6 return 31;
	else if @month < 12 or dbo.jleap(@year) = 1 return 30;
	
	return 29;
end
go

create function dbo.jparse (@year int, @month int, @day int, @hour int, @minute int, @second int) returns datetime with encryption
begin
	declare @days bigint;
	declare @periods int;

	-- Checking accuracy of year
	if @year < 1000 begin
		-- raiserror ('Jalali Parse (value:year) year must be greater than 1000.', 18, 1, 'dbo.jparse')
		return null
	end

	-- Checking accuracy of month
	if @month < 1 or @month > 12 begin
		-- raiserror ('Jalali Parse (value:month) month must be between 1 and 12.', 18, 1, 'dbo.jparse')
		return null
	end

	-- Checking accuracy of day
	if @day < 1 or @day > dbo.jmax(@year, @month) begin
		-- raiserror ('Jalali Parse (value:day) day must be between 1 and 29/30/31.', 18, 1, 'dbo.jparse')
		return null
	end

	-- Calculating amount of total days
	-- -1 because amount of days before requested day
	set @days = -1;

	-- JalaliCriterion = -73
	-- Going to first point of 2820 scale
	set @year += 73;
    
	-- set year posetive
	if @year < 0 begin
		set @periods = ceiling(abs(@year) / 2820.0);
		set @year += @periods * 2820;
		-- DaysPer2820Years = 1029983 days
		set @days -= @periods * 1029983;
	end

	-- DaysPer2820Years = 1029983 days
	-- Calculating amount of years (in 2820 scale)
	set @days += 1029983 * floor(@year / 2820);
	set @year %= 2820;
	-- Calculating first 37 period in the begining of 2820 period
	if @year > 37 begin
		-- DaysPer37Years = 13514
		-- picking up first 37 period in the begining of 2820 period
		set @days += 13514;
		set @year -= 37;

		-- Calculating first 29 period in the begining of 2820 period
		if @year > 29 begin
			-- DaysPer29Years = 10592 days
			-- picking up first 29 period in the begining of 2820 period
			set @days = @days + 10592;
			set @year = @year - 29;

			-- DaysPer128Years = 46751 days
			-- Calculating amount of years (in 128 scale)
			set @days = @days + 46751 * floor(@year / 128);
			set @year = @year % 128;

			-- DaysPer33Years = 12053
			-- Calculating amount of years (in 33 scale)
			set @days = @days + 12053 * floor(@year / 33);
			set @year = @year % 33;
            
		end
	end
    
	if @year > 0 begin
		-- DaysPer4Years = 1461
		-- Calculating amount of years (in 4 scale)
		set @days = @days + 1461 * floor(@year / 4);
		set @year = @year % 4;
        -- DaysPerYear = 365 days
		-- remaining years
		set @days = @days + (365 * @year);
	end

	-- Criterion of Java Date = 1421 (one year more) = 519009 days
	-- Criterion of Java Date + Christian to Jalali
	-- diference between 1970 (UNIX Time first year) and first year of jalali 2820 period = 519009 days
	-- The diference between 1 jan 1970 and 1 far = 286 days
	-- 519009 + 286 = 519295
	set @days = @days - 519295;

	-- Adding month
	if @month < 7
		set @days = @days + (@month - 1) * 31;
	else
		set @days = @days + 186 + (@month - 7) * 30;

	-- Adding days
	set @days = @days + @day;
    
    -- CecoundPerDay = 60 * 60 * 24 = 86400
	-- Calculating amount of total second and returning result
	declare @result datetime;
	set @result = {d '1970-01-01'};

	set @result = dateadd(day, @days, @result);
	set @result = dateadd(hour, @hour, @result);
	set @result = dateadd(minute, @minute, @result);
	set @result = dateadd(second, @second, @result);

    return @result;

end
go

create function dbo.jalaliw(@gregorian datetime) returns int with encryption
begin
	declare @jleap bit;
	declare @jyear int;
	declare @jmonth int;
	declare @jweek int;
	declare @jday int;
	
	declare @num int;
	declare @days int;

	-- JalaliCriterion = -73
	-- First year of 2820 Jalali period
	set @jyear = -73;

	-- diference between 1970 (UNIX Time first year) and first year of jalali 2820 period = 519009 days
	-- The diference between 1 jan 1970 and 1 far = 286 days
	-- 519009 + 286 = 519295
	set @days = floor(
		datediff(day, 
			{d '1970-01-01'}, 
			dateadd(second, 
				- datediff(second, getdate(), getutcdate()), -- Local Time Offset
				@gregorian
				)
			)
		) + 519295;

	-- Calculate week
	set @jweek = (@days + 5) % 7;
	if @jweek < 0 set @jweek = @jweek + 7;

	-- DaysPer2820Years = 1029983 days
	-- Years * (number of 2820 periods)
	set @jyear = @jyear + floor(@days / 1029983) * 2820;
	-- DaysPer2820Years = 1029983
	-- Number of days in each 2820 period
	set @days = @days % 1029983;

	-- DaysPer37Years = 13514
	if @days > 13514 begin
		-- picking up first 37 period at beginning of 2820 period
		set @jyear = @jyear + 37;
		-- DaysPer37Years = 13514
		-- Deducing of days in this period
		set @days = @days - 13514;
        
		-- DaysPer29Years = 10592
		if @days > 10592 begin
			-- picking up first 29 period at beginning of 2820 period
			set @jyear = @jyear + 29;
			-- DaysPer29Years = 10592
			-- Deducing of days in this period
			set @days = @days - 10592;
			
			-- DaysPer128Years = 46751 days
			-- Years * (number of 128 periods)
			set @jyear = @jyear + floor(@days / 46751) * 128;
			-- DaysPer128Years = 46751
			-- Number of days in each 128 period
			set @days = @days % 46751;

			-- DaysPer33Years = 12053
			-- Number of 33 periods
			set @jyear = @jyear + floor(@days / 12053) * 33;
			-- DaysPer33Years = 12053
			-- Number of days in each 4 period
			set @days = @days % 12053;

		end
    end

	-- DaysPer4Years = 1461
	-- Number of 4 periods
	set @jyear = @jyear + floor(@days / 1461) * 4;
	-- DaysPer4Years = 1461
	-- Number of days in each 4 period
	set @days = @days % 1461;
            
	set @jleap = 0;
    
	-- DaysPerYear = 365
	if @days >= 365 begin
		-- DaysPerYear = 365
		-- Number of years
		set @num = floor(@days / 365);
		set @jyear = @jyear + @num;
		-- DaysPerYear = 365
		-- Number of days in these years
		set @days = @days % 365;

		-- Last day in a leap year (12/30)
		if @num = 4 begin
			set @jyear = @jyear - 1;
			-- DaysPerYear = 365
			-- Rolling back the last year
			set @days = @days + 365;
			-- This year is leap
			set @jleap = 1;

		-- Find leap years
		end else if @num = 3 set @jleap = 1

	end

	-- Month and Day
	set @jmonth = 1;
	set @jday = 1;
	if @days < 186 begin
		set @jmonth = @jmonth + floor(@days / 31);
		set @days = @days % 31;
	end else begin
		set @days = @days - 186;
		set @jmonth = @jmonth + 6 + floor(@days / 30);
		set @days = @days % 30;
	end
	
	set @jday = @jday + @days;
	
	return 
		@jday + -- two digits
		@jweek * 100 + -- one deigit
		@jmonth * 1000 + -- two digits
		@jyear * 100000 + -- four digits
		@jleap * 1000000000;
end
go

create function dbo.jalali(@gregorian datetime) returns int with encryption
begin
	declare @jleap bit;
	declare @jyear int;
	declare @jmonth int;
	declare @jday int;
	
	declare @num int;
	declare @days int;

	-- JalaliCriterion = -73
	-- First year of 2820 Jalali period
	set @jyear = -73;

	-- diference between 1970 (UNIX Time first year) and first year of jalali 2820 period = 519009 days
	-- The diference between 1 jan 1970 and 1 far = 286 days
	-- 519009 + 286 = 519295
	set @days = floor(
		datediff(day, 
			{d '1970-01-01'}, 
			dateadd(second, 
				- datediff(second, getdate(), getutcdate()), -- Local Time Offset
				@gregorian
				)
			)
		) + 519295;

	-- DaysPer2820Years = 1029983 days
	-- Years * (number of 2820 periods)
	set @jyear = @jyear + floor(@days / 1029983) * 2820;
	-- DaysPer2820Years = 1029983
	-- Number of days in each 2820 period
	set @days = @days % 1029983;

	-- DaysPer37Years = 13514
	if @days > 13514 begin
		-- picking up first 37 period at beginning of 2820 period
		set @jyear = @jyear + 37;
		-- DaysPer37Years = 13514
		-- Deducing of days in this period
		set @days = @days - 13514;
        
		-- DaysPer29Years = 10592
		if @days > 10592 begin
			-- picking up first 29 period at beginning of 2820 period
			set @jyear = @jyear + 29;
			-- DaysPer29Years = 10592
			-- Deducing of days in this period
			set @days = @days - 10592;
			
			-- DaysPer128Years = 46751 days
			-- Years * (number of 128 periods)
			set @jyear = @jyear + floor(@days / 46751) * 128;
			-- DaysPer128Years = 46751
			-- Number of days in each 128 period
			set @days = @days % 46751;

			-- DaysPer33Years = 12053
			-- Number of 33 periods
			set @jyear = @jyear + floor(@days / 12053) * 33;
			-- DaysPer33Years = 12053
			-- Number of days in each 4 period
			set @days = @days % 12053;

		end
    end

	-- DaysPer4Years = 1461
	-- Number of 4 periods
	set @jyear = @jyear + floor(@days / 1461) * 4;
	-- DaysPer4Years = 1461
	-- Number of days in each 4 period
	set @days = @days % 1461;
            
	set @jleap = 0;
    
	-- DaysPerYear = 365
	if @days >= 365 begin
		-- DaysPerYear = 365
		-- Number of years
		set @num = floor(@days / 365);
		set @jyear = @jyear + @num;
		-- DaysPerYear = 365
		-- Number of days in these years
		set @days = @days % 365;

		-- Last day in a leap year (12/30)
		if @num = 4 begin
			set @jyear = @jyear - 1;
			-- DaysPerYear = 365
			-- Rolling back the last year
			set @days = @days + 365;
			-- This year is leap
			set @jleap = 1;

		-- Find leap years
		end else if @num = 3 set @jleap = 1

	end

	-- Month and Day
	set @jmonth = 1;
	set @jday = 1;
	if @days < 186 begin
		set @jmonth = @jmonth + floor(@days / 31);
		set @days = @days % 31;
	end else begin
		set @days = @days - 186;
		set @jmonth = @jmonth + 6 + floor(@days / 30);
		set @days = @days % 30;
	end
	
	set @jday = @jday + @days;
	
	return 
		@jday + -- two digits
		@jmonth * 100 + -- two digits
		@jyear * 10000; -- four digits
end
go

create function dbo.jyear(@gregorian datetime) returns int with encryption
begin
	declare @jyear int;
	declare @num int;
	declare @days int;

	-- JalaliCriterion = -73
	-- First year of 2820 Jalali period
	set @jyear = -73;

	-- SecondPerDay = 86400
	-- diference between 1970 (UNIX Time first year) and first year of jalali 2820 period = 519009 days
	-- The diference between 1 jan 1970 and 1 far = 286 days
	-- 519009 + 286 = 519295
	set @days = floor(
		datediff(day, 
			{d '1970-01-01'}, 
			dateadd(second, 
				- datediff(second, getdate(), getutcdate()), -- Local Time Offset
				@gregorian
				)
			)
		) + 519295;

	-- DaysPer2820Years = 1029983 days
	-- Years * (number of 2820 periods)
	set @jyear = @jyear + floor(@days / 1029983) * 2820;
	-- DaysPer2820Years = 1029983
	-- Number of days in each 2820 period
	set @days = @days % 1029983;

	-- DaysPer37Years = 13514
	if @days > 13514 begin
		-- picking up first 37 period at beginning of 2820 period
		set @jyear = @jyear + 37;
		-- DaysPer37Years = 13514
		-- Deducing of days in this period
		set @days = @days - 13514;
        
		-- DaysPer29Years = 10592
		if @days > 10592 begin
			-- picking up first 29 period at beginning of 2820 period
			set @jyear = @jyear + 29;
			-- DaysPer29Years = 10592
			-- Deducing of days in this period
			set @days = @days - 10592;
			
			-- DaysPer128Years = 46751 days
			-- Years * (number of 128 periods)
			set @jyear = @jyear + floor(@days / 46751) * 128;
			-- DaysPer128Years = 46751
			-- Number of days in each 128 period
			set @days = @days % 46751;

			-- DaysPer33Years = 12053
			-- Number of 33 periods
			set @jyear = @jyear + floor(@days / 12053) * 33;
			-- DaysPer33Years = 12053
			-- Number of days in each 4 period
			set @days = @days % 12053;

		end
    end

	-- DaysPer4Years = 1461
	-- Number of 4 periods
	set @jyear = @jyear + floor(@days / 1461) * 4;
	-- DaysPer4Years = 1461
	-- Number of days in each 4 period
	set @days = @days % 1461;
    
	-- DaysPerYear = 365
	if @days >= 365 begin
		-- DaysPerYear = 365
		-- Number of years
		set @num = floor(@days / 365);
		set @jyear = @jyear + @num;
		-- Last day in a leap year (12/30)
		if @num = 4 begin
			set @jyear = @jyear - 1;
		end
	end
	
	return @jyear;
end
go

create function jmonth(@gregorian datetime) returns int with encryption
begin	
	declare @num int;
	declare @days int;

	-- SecondPerDay = 86400
	-- diference between 1970 (UNIX Time first year) and first year of jalali 2820 period = 519009 days
	-- The diference between 1 jan 1970 and 1 far = 286 days
	-- 519009 + 286 = 519295
	set @days = floor(
		datediff(day, 
			{d '1970-01-01'}, 
			dateadd(second, 
				- datediff(second, getdate(), getutcdate()), -- Local Time Offset
				@gregorian
				)
			)
		) + 519295;

	-- DaysPer2820Years = 1029983
	-- Number of days in each 2820 period
	set @days = @days % 1029983;
	-- DaysPer37Years = 13514
	if @days > 13514 begin
		-- DaysPer37Years = 13514
		-- Deducing of days in this period
		set @days = @days - 13514;
		-- DaysPer29Years = 10592
		if @days > 10592 begin
			-- DaysPer29Years = 10592
			-- Deducing of days in this period
			set @days = @days - 10592;
			-- DaysPer128Years = 46751
			-- Number of days in each 128 period
			set @days = @days % 46751;
			-- DaysPer33Years = 12053
			-- Number of days in each 4 period
			set @days = @days % 12053;
		end
    end
	-- DaysPer4Years = 1461
	-- Number of days in each 4 period
	set @days = @days % 1461;    
	-- DaysPerYear = 365
	if @days >= 365 begin
		-- DaysPerYear = 365
		-- Number of years
		set @num = floor(@days / 365);
		-- DaysPerYear = 365
		-- Number of days in these years
		set @days = @days % 365;
		-- Last day in a leap year (12/30)
		if @num = 4 begin
			-- DaysPerYear = 365
			-- Rolling back the last year
			set @days = @days + 365;
		end
	end

	if @days < 186
		return 1 + floor(@days / 31);
	else begin
		set @days = @days - 186;
		return 7 + floor(@days / 30);
	end
	
	return 0;
end
go

create function jday(@gregorian datetime) returns int with encryption
begin	
	declare @num int;
	declare @days int;

	-- SecondPerDay = 86400
	-- diference between 1970 (UNIX Time first year) and first year of jalali 2820 period = 519009 days
	-- The diference between 1 jan 1970 and 1 far = 286 days
	-- 519009 + 286 = 519295
	set @days = floor(
		datediff(day, 
			{d '1970-01-01'}, 
			dateadd(second, 
				- datediff(second, getdate(), getutcdate()), -- Local Time Offset
				@gregorian
				)
			)
		) + 519295;

	-- DaysPer2820Years = 1029983
	-- Number of days in each 2820 period
	set @days = @days % 1029983;
	-- DaysPer37Years = 13514
	if @days > 13514 begin
		-- DaysPer37Years = 13514
		-- Deducing of days in this period
		set @days = @days - 13514;
		-- DaysPer29Years = 10592
		if @days > 10592 begin
			-- DaysPer29Years = 10592
			-- Deducing of days in this period
			set @days = @days - 10592;
			-- DaysPer128Years = 46751
			-- Number of days in each 128 period
			set @days = @days % 46751;
			-- DaysPer33Years = 12053
			-- Number of days in each 4 period
			set @days = @days % 12053;
		end
    end
	-- DaysPer4Years = 1461
	-- Number of days in each 4 period
	set @days = @days % 1461;
	-- DaysPerYear = 365
	if @days >= 365 begin
		-- DaysPerYear = 365
		-- Number of years
		set @num = floor(@days / 365);
		-- DaysPerYear = 365
		-- Number of days in these years
		set @days = @days % 365;
		-- Last day in a leap year (12/30)
		if @num = 4 begin
			-- DaysPerYear = 365
			-- Rolling back the last year
			set @days = @days + 365;
		end
	end
    
	if @days < 186
		set @days = @days % 31;
	else begin
		set @days = @days - 186;
		set @days = @days % 30;
	end
	
	return 1 + @days;
end
go

create function jstring(@jdate int, @output nvarchar(max)) returns nvarchar(max) with encryption
begin
	-- y: year
    -- m: month as number
    -- M: monath as string
    -- W: day in week
    -- d: day as number
    
	declare @jyear int;
	declare @jnyear varchar(4);
	declare @jmonth int;
	declare @jnmonth varchar(2);
	declare @jsmonth nvarchar(8);
	declare @jweek int;
	declare @jsweek nvarchar(8);
	declare @jday int;
	declare @jnday varchar(2);
    
	set @jyear = floor(@jdate / 100000) % 10000;
	set @jmonth = floor(@jdate / 1000) % 100;
	set @jweek = floor(@jdate / 100) % 10;
	set @jday = @jdate % 100;

	set @jsmonth = case @jmonth
		when 1 then N'فروردین'
		when 2 then N'اردیبهشت'
		when 3 then N'خرداد'
		when 4 then N'تیر'
		when 5 then N'مرداد'
		when 6 then N'شهریور'
		when 7 then N'مهر'
		when 8 then N'آبان'
		when 9 then N'آذر'
		when 10 then N'دی'
		when 11 then N'بهمن'
		when 12 then N'اسفند'
		else 'none'
		end
    
	set @jsweek = case @jweek
		when 0 then N'شنبه'
		when 1 then N'یکشنبه'
        when 2 then N'دوشنبه'
        when 3 then N'سه‌شنبه'
        when 4 then N'چهارشنبه'
        when 5 then N'پنجشنبه'
        when 6 then N'جمعه'
        else ''
        end
    
	set @jnyear = @jyear;
    while len(@jnyear) < 4 begin set @jnyear = concat('0', @jnyear); end
	set @jnmonth = @jmonth;
    while len(@jnmonth) < 2 begin set @jnmonth = concat('0', @jnmonth); end
	set @jnday = @jday;
    while len(@jnday) < 2 begin set @jnday = concat('0', @jnday); end
    
	set @output = REPLACE(@output,'y',@jnyear);
	set @output = REPLACE(@output,'mm',@jnmonth);
	set @output = REPLACE(@output,'m',@jmonth);
	set @output = REPLACE(@output,'M',@jsmonth);
	set @output = REPLACE(@output,'W',@jsweek);
	set @output = REPLACE(@output,'dd',@jnday);
	set @output = REPLACE(@output,'d',@jday);
	
	set @output = REPLACE(@output,'0',N'۰');
	set @output = REPLACE(@output,'1',N'۱');
	set @output = REPLACE(@output,'2',N'۲');
	set @output = REPLACE(@output,'3',N'۳');
	set @output = REPLACE(@output,'4',N'۴');
	set @output = REPLACE(@output,'5',N'۵');
	set @output = REPLACE(@output,'6',N'۶');
	set @output = REPLACE(@output,'7',N'۷');
	set @output = REPLACE(@output,'8',N'۸');
	set @output = REPLACE(@output,'9',N'۹');
    
    return @output;
end
go

create function jprint(@jdate int, @output nvarchar(max), @outmode varchar(2)) returns nvarchar(max) with encryption
begin
	-- y: year
    -- m: month as number
    -- M: monath as string
    -- W: day in week
    -- d: day as number
    
	declare @jyear int;
	declare @jnyear varchar(4);
	declare @jmonth int;
	declare @jnmonth varchar(2);
	declare @jsmonth varchar(8);
	declare @jweek int;
	declare @jsweek varchar(8);
	declare @jday int;
	declare @jnday varchar(2);
    
	set @jyear = floor(@jdate / 100000) % 10000;
	set @jmonth = floor(@jdate / 1000) % 100;
	set @jweek = floor(@jdate / 100) % 10;
	set @jday = @jdate % 100;

	set @jsmonth = case @jmonth
		when 1 then N'فروردین'
		when 2 then N'اردیبهشت'
		when 3 then N'خرداد'
		when 4 then N'تیر'
		when 5 then N'مرداد'
		when 6 then N'شهریور'
		when 7 then N'مهر'
		when 8 then N'آبان'
		when 9 then N'آذر'
		when 10 then N'دی'
		when 11 then N'بهمن'
		when 12 then N'اسفند'
		else 'none'
		end
    
	set @jsweek = case @jweek
		when 0 then N'شنبه'
		when 1 then N'یکشنبه'
        when 2 then N'دوشنبه'
        when 3 then N'سه‌شنبه'
        when 4 then N'چهارشنبه'
        when 5 then N'پنجشنبه'
        when 6 then N'جمعه'
        else ''
        end
        
	set @jnyear = @jyear;
    while len(@jnyear) < 4 begin set @jnyear = concat('0', @jnyear); end
	set @jnmonth = @jmonth;
    while len(@jnmonth) < 2 begin set @jnmonth = concat('0', @jnmonth); end
	set @jnday = @jday;
    while len(@jnday) < 2 begin set @jnday = concat('0', @jnday); end
    
	set @output = REPLACE(@output,'y',@jnyear);
	set @output = REPLACE(@output,'mm',@jnmonth);
	set @output = REPLACE(@output,'m',@jmonth);
	set @output = REPLACE(@output,'M',@jsmonth);
	set @output = REPLACE(@output,'W',@jsweek);
	set @output = REPLACE(@output,'dd',@jnday);
	set @output = REPLACE(@output,'d',@jday);
	
    if charindex('p', @outmode) > 0 begin
		set @output = REPLACE(@output,'0',N'۰');
		set @output = REPLACE(@output,'1',N'۱');
		set @output = REPLACE(@output,'2',N'۲');
		set @output = REPLACE(@output,'3',N'۳');
		set @output = REPLACE(@output,'4',N'۴');
		set @output = REPLACE(@output,'5',N'۵');
		set @output = REPLACE(@output,'6',N'۶');
		set @output = REPLACE(@output,'7',N'۷');
		set @output = REPLACE(@output,'8',N'۸');
		set @output = REPLACE(@output,'9',N'۹');
    end
    
    return @output;
end
go

declare @custom datetime; 
set @custom = {d '2002-02-20'};
select
	@custom as 'Custom',
	dbo.jparse(dbo.jyear(@custom), dbo.jmonth(@custom), dbo.jday(@custom), 0, 0, 0) as 'Parse',
	dbo.jalali(@custom) as 'Number Jalali',
	dbo.jstring(dbo.jalali(@custom), 'W d M y') as 'String Jalali',
	dbo.jprint(dbo.jalali(@custom), 'y/m/d', 'n') as 'Digital String Jalali';
    
select
	getdate() as 'Now',
	dbo.jparse(dbo.jyear(getdate()), dbo.jmonth(getdate()), dbo.jday(getdate()), 0, 0, 0) as 'Parse',
	dbo.jalali(getdate()) as 'Number Jalali',
	dbo.jstring(dbo.jalali(getdate()), 'W d M y') as 'String Jalali',
	dbo.jprint(dbo.jalali(getdate()), 'y/m/d', 'n') as 'Digital String Jalali';