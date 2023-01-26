import { ComponentFixture, TestBed } from '@angular/core/testing';

import { HistoricalTransactionsComponent } from './historical-transactions.component';

describe('HistoricalTransactionsComponent', () => {
  let component: HistoricalTransactionsComponent;
  let fixture: ComponentFixture<HistoricalTransactionsComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ HistoricalTransactionsComponent ]
    })
    .compileComponents();

    fixture = TestBed.createComponent(HistoricalTransactionsComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
