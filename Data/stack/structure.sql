use RahavardNovin3;
/*
alter table Trade add RecordType char(1) null;
GO

CREATE NONCLUSTERED INDEX [IX_Trade_RecordType]
	ON [Trade] ([RecordType] ASC) INCLUDE([InstrumentID])
GO

CREATE NONCLUSTERED INDEX [IX_Trade_InstrumentID_RecordType_DateTimeEn]
	ON [Trade]([InstrumentID] ASC, [RecordType] ASC, [DateTimeEn] DESC) INCLUDE([ClosePrice],[ClosePriceChange])
GO

create table Pattern (
	Section int not null,
	Ranking int not null,
	primary key (Section, Ranking)
)
GO*/