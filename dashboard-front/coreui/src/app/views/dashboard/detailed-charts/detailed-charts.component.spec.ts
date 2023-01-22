import { ComponentFixture, TestBed } from '@angular/core/testing';

import { DetailedChartsComponent } from './detailed-charts.component';

describe('DetailedChartsComponent', () => {
  let component: DetailedChartsComponent;
  let fixture: ComponentFixture<DetailedChartsComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ DetailedChartsComponent ]
    })
    .compileComponents();

    fixture = TestBed.createComponent(DetailedChartsComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
