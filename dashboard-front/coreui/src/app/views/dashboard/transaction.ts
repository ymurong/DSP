
import { HateoasResource, Resource } from '@lagoshny/ngx-hateoas-client';


export class Transaction {
   
    merchant: string = "";
    card_schema: string = "";
    is_credit: boolean = false;
    eur_amount: Number = 0;
    ip_country: string = "";
    issuing_country: string = "";
    device_type: string = "";
    ip_address: string = "";
    email_address: string = "";
    card_number: string = "";
    shopper_interaction: string = "";
    zip_code: string = "";
    card_bin: string = "";
    has_fraudulent_dispute: boolean = false;
    is_refused_by_adyen: boolean = false;
    created_at!: Date;
    updated_at!: Date;
    psp_reference: bigint = BigInt(999999999);


    constructor(values: object = {}) {
        Object.assign(this as any, values);
      }
}