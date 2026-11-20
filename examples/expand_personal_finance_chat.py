"""Expand the original finance dialogue corpus with deterministic examples."""

from pathlib import Path


OUTPUT = Path(__file__).with_name("personal_finance_chat.txt")
MARKER = "\n# Additional scenario-based practice dialogues\n"

TOPICS = [
    (
        "Planning a first budget",
        "A useful budget begins with observed take-home income and real expenses, not idealized guesses. Separate essentials, debt obligations, savings goals, and flexible spending. Review a few weeks of transactions, make a plan that leaves some room for enjoyment, and revise it after comparing the plan with reality. A budget is a decision aid, not a measure of personal worth.",
        "What should I track first?",
        "For one month, record the date, amount, and broad category for each purchase and bill. Do not try to change every habit at once. At the end, compare the totals with your take-home pay, mark any irregular costs, and choose one adjustment that feels practical. A simple notes app or spreadsheet is enough; protect it like other personal information.",
        [
            ("I have never tracked spending and worry that a budget will make me feel guilty.", "Start with observation rather than restriction. Track a month without labeling purchases as good or bad; the purpose is to learn where money actually goes.", "Begin with just three groups: fixed bills, everyday spending, and savings or debt. Add detail only when it helps answer a real question."),
            ("My partner and I disagree about whether a budget means we cannot have fun.", "Agree on shared priorities and include a realistic personal-spending amount for each person. A plan that allows reasonable enjoyment is more likely to last.", "Each person can privately list priorities, then compare the overlap and tradeoffs together. Review the plan at a calm, scheduled time rather than during an argument."),
            ("My expenses change a lot from month to month.", "Separate steady bills from variable and occasional costs. Use several months of records where available so one unusually cheap month does not become the plan.", "List annual or irregular expenses and divide their expected cost across the months before they are due. Update the estimate when you learn the actual amount."),
            ("I have a low income and the usual budgeting percentages do not fit.", "Popular percentage rules are only rough examples. Cover immediate essentials and obligations first, then use the remaining amount according to your circumstances and local support options.", "Write down income and essential costs using actual amounts. If essentials exceed available income, look for reputable local benefits or nonprofit support rather than blaming yourself for a percentage mismatch."),
        ],
    ),
    (
        "Irregular income and cash flow",
        "When income varies, planning from a high-earning month can create a shortfall later. A conservative estimate based on reliable history is safer for routine commitments. Prioritize necessities, required payments, and known upcoming bills. In stronger months, set aside money for leaner periods before increasing discretionary spending. The method should reflect the person's actual pay schedule and obligations.",
        "How can I handle a month when income falls below my estimate?",
        "Recalculate using money already available and confirmed income, then prioritize food, housing, utilities, transport needed for work, and required obligations. Contact providers early if a bill may be late and ask about legitimate hardship or payment options. Avoid making a promise you cannot keep, and check local advice resources if the gap is recurring.",
        [
            ("I am paid by freelance projects, so I cannot predict the next paycheck.", "Treat signed, confirmed work differently from hoped-for work. Plan recurring expenses around a cautious income baseline and track invoices, due dates, and actual deposits.", "Make a simple cash-flow calendar with expected invoice dates and bill due dates. Keep business and household money clearly recorded, and set aside applicable taxes according to local rules."),
            ("My hours are cut in winter every year.", "A recurring seasonal dip is a known expense of the job. Estimate the low-season gap from past records and reserve part of higher-season income if your budget allows.", "Look back at a full year of income and bills. Choose a monthly reserve amount that is feasible now, and identify which optional costs could be paused during the slower season."),
            ("A large payment arrived, but I have several months of bills ahead.", "A large deposit can feel like extra spending money even when it must cover future work gaps. Assign it to upcoming obligations and goals before treating any remainder as flexible.", "List the bills and essential costs due before the next reliable income. Keep the corresponding amount accessible and track what remains after each payment."),
            ("I am self-employed and sometimes forget that a busy month is not a normal month.", "Use a rolling view of income across several months and separate business receipts from personal take-home money. Taxes, supplies, and slow periods may all need a reserve.", "Review a few months of bank and invoice records, then set a conservative transfer to personal spending. Ask a qualified local tax professional about obligations that apply to your work."),
        ],
    ),
    (
        "Building an emergency buffer",
        "An emergency reserve is intended for unplanned needs or interruptions, not predictable bills that can be planned for separately. Start with an achievable amount and build gradually. The right size depends on income stability, household responsibilities, insurance, and access to other support. Keep the reserve accessible, understand account terms, and replenish it after a legitimate use when possible.",
        "How do I decide what counts as an emergency?",
        "Ask whether the expense is necessary, unexpected, and time-sensitive, and whether delaying it would create a larger problem. A planned annual fee is usually better handled by saving for it ahead of time; an urgent repair needed to get to work may be a reasonable reserve use. Your circumstances can change the answer.",
        [
            ("I can only save a very small amount each payday.", "A small, regular contribution can still establish the habit and gradually create options. Do not compare your starting point with someone else's target.", "Choose an amount that does not cause missed essentials or required payments. Automate it only if the account balance and pay timing make that safe, and review it when circumstances change."),
            ("I used my emergency savings for an urgent car repair and feel like I failed.", "Using a reserve for an actual urgent need is what it is for. Rebuilding can happen gradually; the repair may also have prevented a larger cost or work disruption.", "Record the repair as a one-time event and restart contributions at a manageable pace. If similar repairs keep occurring, create a separate maintenance category for predictable upkeep."),
            ("Should my emergency fund be in cash at home?", "Physical cash can be immediately available but may be lost, stolen, or damaged and may earn no interest. Account access, deposit protections, and risks vary by location.", "Compare safe, accessible options available where you live. Avoid keeping more physical cash than you can reasonably secure, and do not share account credentials with anyone."),
            ("My income is stable, but I have several people depending on me.", "Household responsibilities can increase the consequences of an interruption. Consider essential monthly costs, insurance, support networks, and the time it could take to replace income.", "Estimate the essential costs you would need to cover for a period that feels relevant to your household, then build toward it in stages. Revisit the estimate after major life changes."),
        ],
    ),
    (
        "Prioritizing debt payments",
        "A debt plan starts with accurate balances, interest rates, minimums, due dates, fees, and any promotional terms. Required payments help avoid preventable consequences. After essentials and a workable cash buffer, paying extra toward costly debt may reduce interest, while smaller balances may offer motivational wins. Compare the tradeoffs using the actual contracts rather than assuming one strategy suits everyone.",
        "Should I pay the smallest balance first or the highest-rate balance first?",
        "The highest-rate-first approach can reduce interest cost when all required payments are maintained. The smallest-balance-first approach may create visible progress that helps some people stay consistent. List the numbers, estimate the difference, and choose a method you can sustain; either method is different from missing minimums.",
        [
            ("I have four cards and feel overwhelmed by the total.", "Make a private inventory of each balance, rate, minimum, and due date before choosing a strategy. Seeing the whole picture is uncomfortable for some people but prevents decisions based on incomplete information.", "Gather current statements and put the debts in one table. Set reminders for minimums, then choose one extra-payment target only after essentials and near-term bills are covered."),
            ("One debt has a promotional low rate that ends later this year.", "Check the exact end date and what rate applies afterward. Promotional terms can differ, and some agreements have conditions or deferred-interest features that are important to understand.", "Read the agreement and statement, note the deadline, and calculate what balance might remain. If the terms are unclear, contact the provider through an official channel before relying on an assumption."),
            ("A collector contacted me about a debt I do not recognize.", "Do not rush to pay or disclose sensitive information to an unexpected caller. Rules for debt validation and dispute deadlines vary by location, so verify the collector and the account through official written channels.", "Save the communication, independently find the relevant consumer-protection agency or legal-aid resource for your area, and follow its instructions. Avoid using contact details provided only in a suspicious message."),
            ("I can make the minimum payments but have no extra money this month.", "Maintaining essentials and required payments is the immediate priority. A month without extra principal payment does not erase prior progress.", "Review whether any upcoming bill can be discussed with its provider, and make a realistic plan for next month. Do not take a new high-cost loan merely to preserve an aggressive payoff schedule without understanding the full cost."),
        ],
    ),
    (
        "Credit cards and payment habits",
        "A credit card is a payment and borrowing tool, not additional income. Understand the statement date, due date, interest terms, fees, and what happens when only a minimum is paid. Paying the statement balance in full by the due date may avoid purchase interest when the agreement provides a grace period; cash advances and other transactions can have different terms. Local credit-reporting rules vary.",
        "Do I need to carry a balance to build credit?",
        "In many credit-scoring systems, paying interest by carrying a balance is not required to build a payment history. The effect of account age, utilization, and payment reporting depends on local rules and scoring models. Use credit only when repayment fits your plan, not to chase an uncertain score benefit.",
        [
            ("I use my card for groceries but sometimes forget the due date.", "A missed due date can lead to fees or other consequences under the agreement. A reminder or automatic payment may help, but autopay should be set with awareness of the linked account balance.", "Set a calendar alert several days before the due date and check the statement total. If using autopay, confirm the amount, date, and that enough money will be available."),
            ("The minimum payment looks affordable, so can I just pay that?", "The minimum keeps the account from being treated as unpaid under many agreements, but it can leave a balance accruing interest and take much longer to repay. Check the statement's payoff examples and interest rate.", "If possible, choose a payment above the minimum that still leaves room for essentials. Avoid promising a fixed amount until you have checked upcoming bills and the agreement."),
            ("I am thinking about closing a card I no longer use.", "Consider annual fees, fraud exposure, account terms, and possible effects on utilization or credit history in the scoring system where you live. There is no single answer for every account.", "Review the fee schedule and account security options. If you close it, use the issuer's official process, confirm the balance is zero, and retain confirmation."),
            ("A retailer offers a discount if I open a store card at checkout.", "Compare the immediate discount with the card's interest rate, fees, credit inquiry, and your ability to pay in full. A discount can cost more than it saves if it leads to interest or unplanned purchases.", "Pause long enough to read the key terms and ask whether you would still want the account without the discount. You can decline without needing to justify the decision."),
        ],
    ),
    (
        "Understanding loan offers",
        "Compare loans using the annual percentage rate or equivalent local cost measure, fees, term, payment schedule, total repayment, variable-rate conditions, and penalties. A lower monthly payment may result from a longer term and higher total interest. Advertised rates may depend on eligibility. Read the actual written offer and do not rely only on a salesperson's verbal estimate.",
        "What numbers should I compare when shopping for a loan?",
        "Compare the amount borrowed, rate and whether it can change, upfront and ongoing fees, term, required payment, total amount repaid, and consequences of late or early payment. If two offers use different assumptions, ask for a written breakdown on the same loan amount and term.",
        [
            ("The dealer focuses on a monthly payment and says the rate is competitive.", "A payment alone does not reveal the total cost. The term, amount financed, fees, add-ons, and rate all affect it.", "Request the itemized out-the-door price, amount financed, rate, term, and total payments separately. Compare the financing independently from the purchase decision."),
            ("One lender advertises a very low rate but says it depends on approval.", "The advertised rate may be limited to applicants meeting specific criteria. Your actual rate and fees are not known until the lender provides an offer.", "Ask what assumptions qualify for the advertised rate and request your actual written terms before committing. Do not provide sensitive information through an unverified link."),
            ("I want to refinance because the payment could be lower.", "A lower payment can help monthly cash flow but may increase total cost if the new term is much longer or fees are added. Compare remaining payments under the current loan with all payments under the new one.", "Ask for the new payoff amount, fees, rate, term, and total repayment estimate. Check whether the rate is fixed or variable and whether the loan has prepayment conditions."),
            ("I am offered a loan with a variable interest rate.", "A variable rate can change under the agreement, which may change the payment or total cost. Understand the index, adjustment schedule, caps, and what happens if rates rise.", "Ask the lender to show how the payment could change under plausible rate increases. Decide whether your budget could handle those payments before accepting."),
        ],
    ),
    (
        "Planning a vehicle purchase",
        "A vehicle's cost includes more than its advertised price: financing, insurance, fuel or charging, registration, taxes, maintenance, repairs, parking, and depreciation. Get insurance estimates and inspect the vehicle before committing. Compare the total ownership cost with transportation needs and reliable budget capacity, not only the monthly payment or trade-in estimate.",
        "How do I decide whether I can afford a car?",
        "Estimate monthly and one-time costs using quotes and records for the specific vehicle and location. Include a repair buffer and consider how a payment affects other goals. If the estimate only works when nothing goes wrong, the purchase may be too financially fragile.",
        [
            ("I found a used car that costs more than my yearly take-home pay.", "The purchase price alone does not determine affordability, but a large loan relative to income can leave little room for essentials and unexpected costs. Consider less expensive transportation options and the full repayment schedule.", "Write down take-home income, current obligations, insurance quotes, expected maintenance, and the loan's total cost. Decide on a maximum total cost before discussing financing."),
            ("The dealer says my trade-in makes the monthly payment very small.", "A trade-in changes the amount financed, but it does not necessarily make the vehicle inexpensive. Check the trade-in value, any loan balance on it, price, add-ons, rate, and term.", "Ask for the purchase price and trade-in figures separately in writing. Compare the loan offer with financing obtained independently, if available."),
            ("I am choosing between an older car with no loan and a newer car with payments.", "Compare likely repair and maintenance costs, reliability needs, safety, insurance, and the value of keeping cash available. Neither age nor a zero payment guarantees the lowest total cost.", "Get an independent inspection and insurance quotes for each option. Estimate several years of ownership costs using realistic assumptions."),
            ("I need a car for work but worry the loan is a stretch.", "Reliable transport may be important, but a loan that leaves no room for fuel, insurance, repairs, or income changes can create new risk.", "Explore the least costly reliable option and compare public transit, carpooling, and temporary alternatives where feasible. Ask lenders for complete written terms and avoid focusing only on payment size."),
        ],
    ),
    (
        "Everyday saving habits",
        "Savings goals work better when they are specific, visible, and matched to a time horizon. Separate near-term goals from emergency reserves and longer-term investing. An automatic transfer can help if timing and balances are predictable, but it should not trigger overdrafts or interfere with essential bills. Small consistent actions are more useful than an unsustainable target.",
        "How can I save when there is very little left after bills?",
        "First verify the actual amounts and timing of income and essential costs. If there is a small surplus, choose a modest amount and review it after a month. If there is no surplus, a spreadsheet cannot create money; investigate available local assistance or ways to adjust a recurring cost without compromising safety or health.",
        [
            ("I save only when I remember, and then unexpected purchases use it.", "A clearly labeled goal and a routine review may make savings more visible. Keep short-term planned expenses separate from emergency money so they do not compete invisibly.", "Give each savings category a name and target. Set a reminder or a small scheduled transfer only after checking pay timing and minimum account balances."),
            ("My bank offers a savings rate that changes frequently.", "Rates can change and may have eligibility requirements, minimums, fees, or withdrawal limits. Compare the account's full terms and protections, not just its advertised rate.", "Read the current fee and rate disclosures from the institution's official site. Confirm how the account is protected in your jurisdiction and whether access suits your needs."),
            ("I have several goals: travel, a laptop, and emergency savings.", "Separate goals by importance and deadline. Emergency capacity and essential upcoming costs may deserve priority, but personal goals can still be part of a balanced plan.", "Write each target amount and date, then estimate a monthly contribution. If the totals exceed your available surplus, extend a deadline or adjust a goal rather than relying on debt."),
            ("I feel like saving five dollars is pointless.", "A small amount may not solve a large future expense, but it can begin a habit and make progress measurable. The purpose is to build capacity gradually, not to pretend that every income constraint is easy to overcome.", "Choose a repeatable amount, record it, and increase only if income or expenses allow. Also look for structural options such as benefits, bill assistance, or more stable work where relevant."),
        ],
    ),
    (
        "Subscriptions and recurring costs",
        "Recurring charges often continue because they are easy to forget. Review bank and card statements, identify services and renewal dates, and distinguish contracts from subscriptions that can be canceled at any time. Check cancellation confirmation and any remaining balance. Small purchases can be worthwhile; the goal is to ensure they match current priorities.",
        "How can I find subscriptions I forgot about?",
        "Review several months of statements across every card and account used for online purchases. Search email for receipts and renewal notices. Record the service, amount, frequency, next renewal, and cancellation method, then verify that any cancellation actually stopped billing.",
        [
            ("A free trial renews tomorrow, and I am not sure whether I canceled it.", "Do not rely on memory; check the account's subscription page and written confirmation. Cancellation rules and refund rights depend on the provider and local consumer law.", "Use the provider's official site or app rather than a message link. Save the confirmation and monitor the next statement for an unexpected charge."),
            ("I pay for streaming services but use only one most months.", "Consider rotating services or pausing those you rarely use, while checking for bundles, annual terms, or cancellation fees. Entertainment spending is a choice, not automatically a mistake.", "List the services and renewal dates, then cancel or pause one at a time. Set a reminder to reassess before re-subscribing."),
            ("A membership has an annual fee that is coming up.", "Compare the upcoming fee with the benefits you realistically used, not the benefits you intended to use. Check the cancellation deadline and whether membership is tied to other services.", "Read the renewal and cancellation terms in the account. Decide before the deadline and retain confirmation if you cancel."),
            ("I see a small charge I do not recognize on my card.", "Treat unfamiliar transactions as worth checking. It may be a merchant's billing name, a trial, or unauthorized activity; do not assume which without verification.", "Check receipts and household account users, then contact the card issuer using the number on the card or official app if it remains unexplained. Follow its dispute instructions promptly."),
        ],
    ),
    (
        "Shopping and purchase decisions",
        "A pause can separate a real need from pressure, advertising, or a temporary mood. Compare the total price, expected use, maintenance, return terms, and effect on other priorities. A waiting period is especially useful for nonessential purchases. The goal is not to avoid all treats but to spend intentionally and safely.",
        "How do I stop impulse purchases without making life miserable?",
        "Try a cooling-off period for nonessential items, remove saved payment details if they encourage rushed buying, and keep a realistic discretionary category. Notice which situations trigger spending and choose a different routine, such as leaving a shopping app or making a list.",
        [
            ("A flash sale says the price disappears in ten minutes.", "Time pressure is a common sales tactic. A discount saves money only if the item is needed and the total price fits your plan.", "Close the offer and check the usual price, return terms, and whether you would buy it without a countdown. A legitimate purchase can wait until you have verified the details."),
            ("I buy things online when I feel stressed.", "Spending can briefly change a mood while creating later budget stress. Treat the emotional trigger with care rather than shame.", "Try a short delay and a non-purchase activity that helps you decompress. If spending feels uncontrollable or causes serious harm, consider trusted professional or community support."),
            ("Buying in bulk seems cheaper, but I throw some food away.", "Compare unit price with the amount your household can actually use before spoilage. Storage, transport, and cash tied up in unused goods also have costs.", "Track what is wasted for a few weeks and buy a smaller amount next time. Choose bulk only for items you regularly use and can store safely."),
            ("My friends choose expensive restaurants and I feel awkward opting out.", "Social plans can matter, but you can set a spending boundary without judging anyone else's choice. Suggest a lower-cost activity or review the menu before agreeing.", "Decide what you can comfortably spend, then communicate it simply. You do not need to disclose private financial details to decline an expensive plan."),
        ],
    ),
    (
        "Planning for annual and irregular bills",
        "Some expenses are irregular but predictable: insurance renewals, school costs, maintenance, or annual fees. Treating them as surprises can disrupt monthly cash flow. Estimate the timing and amount, divide by the months remaining, and save if possible. Update the estimate when a bill changes and keep true emergencies separate.",
        "How do I prepare for bills that arrive only once or twice a year?",
        "List each bill, due date, expected amount, and payment method. Divide the expected amount by the time remaining, then reserve that portion monthly if your cash flow allows. Confirm the actual bill before paying and keep the reserve accessible until it is due.",
        [
            ("My car registration arrives once a year and always surprises me.", "The due date is irregular but predictable, so it can be planned as a monthly cost. The amount may change, so use the latest official estimate.", "Look up the renewal month and current fee through the official government service. Divide the amount across the months before it is due and set a reminder."),
            ("School supplies and activity fees make the start of term expensive.", "Seasonal costs are easier to manage when listed before the season. Check what can be reused, what is required, and whether the school offers support programs.", "Review last year's costs and ask the school for a current calendar or supply list. Save gradually if possible and verify any assistance through official school channels."),
            ("My insurance premium is due annually, and I am considering monthly installments.", "Monthly billing may improve cash flow but could include fees or a different total cost. Compare payment methods and understand coverage, renewal, and cancellation terms.", "Ask the insurer for the full annual cost under each payment option in writing. Consider affordability and account fees rather than assuming monthly is always cheaper."),
            ("A home repair is not urgent, but I know it will be needed soon.", "A known repair is a planned maintenance goal, not necessarily an emergency. Getting estimates and timing it before failure may reduce disruption, but safety-critical problems need qualified assessment.", "Ask reputable providers for written estimates, check credentials where relevant, and create a savings timeline. Avoid delaying an issue that could make the property unsafe or cause greater damage."),
        ],
    ),
    (
        "Financial goals and tradeoffs",
        "A goal is easier to plan when it has a clear amount, deadline, and reason. Rank goals by urgency and impact, then compare their required contributions with available income. Some goals can be delayed, scaled, or divided into stages. Avoid financing a goal with expensive debt without understanding the full cost and alternative tradeoffs.",
        "I have too many goals and cannot fund them all. What should I do?",
        "Put essential needs, required obligations, and time-sensitive goals first. Estimate the monthly amount each goal needs, then adjust deadlines or amounts if the total exceeds your available surplus. Choosing a sequence is not failure; it is a way to make progress without pretending resources are unlimited.",
        [
            ("I want to travel while also paying down debt.", "Both rest and debt reduction can matter. Compare the debt cost, minimum-payment requirements, travel budget, and whether the trip would create new borrowing.", "Set a travel amount that can be funded in cash without missing essentials or required payments. If that requires a longer timeline, decide whether the revised date still feels worthwhile."),
            ("I need a laptop for work but my current one still functions.", "Estimate how essential the upgrade is, whether repairs or refurbished options meet the need, and how the purchase affects other priorities. Avoid focusing only on a feature list.", "Define the minimum required specifications for your actual work and compare total prices, warranty, and return terms. Set a spending ceiling before browsing."),
            ("My family expects me to contribute money, but it strains my own bills.", "Family support can be important, but it must be weighed against your own essentials and obligations. A clear, sustainable limit is safer than making promises you cannot maintain.", "Decide what amount, if any, fits your plan and explain it respectfully. If direct money is not feasible, consider nonfinancial help without sharing more private information than you want."),
            ("I save for goals but keep changing them when a new idea appears.", "Frequent changes can make progress hard to see. A short review period gives a goal time to work while still allowing changes for genuine new information.", "Write the current goal, target, and review date. Keep a separate list of new ideas and revisit them at the scheduled review rather than changing the plan immediately."),
        ],
    ),
    (
        "Reading a pay statement",
        "A pay statement may show gross pay, deductions, taxes, benefits, and net pay. The exact layout depends on employer and jurisdiction. Comparing statements can reveal a changed withholding, benefit contribution, overtime amount, or deduction, but labels can be unfamiliar. Ask payroll or a qualified local adviser about unclear items instead of guessing.",
        "My take-home pay is lower than I expected. What should I check?",
        "Compare the current statement with a prior one and look at hours, gross pay, tax withholding, benefits, retirement contributions, and other deductions. Check whether the change is temporary or recurring. Do not share full account numbers or identity details when asking for general help.",
        [
            ("My paycheck changed after I enrolled in health benefits.", "Benefit premiums or contribution choices may change net pay, but the exact explanation should appear in enrollment information or payroll records.", "Compare the deduction lines before and after enrollment and contact HR or payroll through a verified channel if the amount does not match your election."),
            ("I worked overtime, but the deposit seems smaller than I expected.", "Gross earnings and net deposit differ because of taxes, benefits, deductions, and possibly pay-period timing. Overtime rules vary by role and location.", "Check the hours and rate shown, compare them with your time records, and ask payroll about a specific discrepancy. Use local labor-agency guidance if the issue remains unresolved."),
            ("I started a new job and do not understand the tax withholding section.", "Withholding is an estimated payment toward tax obligations and is not always the final tax amount. Forms and rules differ by jurisdiction and personal circumstances.", "Use official tax-agency resources for your location or consult a qualified tax professional. Avoid relying on an informal calculator for a complex situation."),
            ("A deduction appears that I did not expect.", "A deduction may be authorized or required, but you should not assume its purpose. Review onboarding documents and the pay statement's explanation.", "Ask payroll for the deduction name, basis, and authorization using a work contact you trust. Keep copies of statements and related correspondence."),
        ],
    ),
    (
        "Bank accounts and fees",
        "Account fees can depend on minimum balances, transaction counts, overdrafts, ATM use, or monthly activity. Account protections and insurance also vary by country and institution. Read the current fee schedule and account terms. A no-fee label may still have conditions, so compare how the account works in ordinary use.",
        "How do I choose a checking account?",
        "Compare monthly fees, balance requirements, ATM access, overdraft settings, transfer timing, customer support, deposit protections, and accessibility. Consider the transactions you actually make and confirm terms directly with the institution before moving money.",
        [
            ("My bank charges a monthly fee unless I keep a minimum balance.", "The fee is part of the account's real cost. Check whether there are legitimate ways to waive it and whether maintaining the minimum would leave too little cash for bills.", "Read the current account terms and compare other institutions available in your area. Plan a safe transition that keeps direct deposits and automatic payments from being disrupted."),
            ("I was charged an overdraft fee after an automatic bill payment.", "Review the transaction timing, overdraft settings, and account agreement. Rules and refund practices vary, and the bank can explain how it processed the payment.", "Contact the bank using its official number, ask about the specific transaction, and request information about alerts or opt-out settings. Keep enough buffer if automatic payments remain enabled."),
            ("A bank text asks me to click a link to verify my account.", "Unexpected links can be phishing even when the message uses a familiar logo or urgent wording. Do not enter credentials through the link.", "Open the official app or type the bank's known address yourself, or call the number on your card. Report the message through the bank's official fraud channel."),
            ("I want to switch banks but have many automatic payments.", "A careful transition avoids missed bills or lost deposits. Keep the old account open long enough to confirm the new one works, subject to fee and balance requirements.", "List direct deposits and recurring debits, update them one at a time, and verify successful transactions. Move remaining funds only after pending payments clear."),
        ],
    ),
    (
        "Avoiding scams and impersonation",
        "Scammers may create urgency, impersonate trusted organizations, or request payment through hard-to-reverse methods. A legitimate institution should not need your password or one-time security code. Verify unexpected requests independently using a trusted contact method, and do not let a caller control your device or pressure you to keep the conversation secret.",
        "Someone says my account will be frozen unless I pay immediately with gift cards.",
        "That payment request is a strong scam warning. Do not buy or share gift-card codes, cryptocurrency, or transfer credentials. End the contact and reach the organization through an official number or website you locate independently.",
        [
            ("A caller claims to be from the government and threatens arrest over an unpaid fee.", "Threats demanding immediate unusual payment are a common impersonation tactic. Government agencies generally have formal processes, but details and procedures vary by jurisdiction.", "Do not use the caller's payment link or callback number. Find the agency's official contact details yourself and ask whether the claim is real."),
            ("A person I met online needs money for an emergency and promises to repay me.", "Emotional urgency and requests for money from someone you have not independently verified can create serious risk. Transfers may be difficult or impossible to reverse.", "Pause, verify the person's identity through another trusted channel, and discuss the request with someone you trust. Never share login or identity codes to prove trust."),
            ("I already gave a one-time passcode to someone who called me.", "Act quickly but do not panic. Contact the affected provider using its official number or app, explain what happened, and follow its security instructions.", "Change credentials from a trusted device if instructed, review recent transactions, secure email and phone accounts, and report unauthorized activity to relevant local authorities or agencies."),
            ("A bank employee asks me to move money to a 'safe account'.", "A request to transfer funds to protect them is a frequent fraud pattern. Do not move money based on an incoming call, even if the caller knows personal details.", "End the call and contact the bank using the phone number on your card. Ask its fraud team whether your account needs action."),
        ],
    ),
    (
        "Planning charitable giving",
        "Giving can be part of a financial plan, but a donation should be voluntary and affordable. Verify the organization, understand how funds are used, and distinguish a one-time gift from a recurring commitment. Be cautious of pressure after disasters or requests for unusual payment methods. Tax treatment depends on local law and eligibility.",
        "How can I tell whether a charity request is legitimate?",
        "Check the organization through an independent registry or reputable local charity evaluator, confirm its official website, and read how donations are used. Do not rely only on a caller's claims, copied logos, or a social-media post. Never share banking passwords or verification codes.",
        [
            ("A fundraiser asks for a recurring monthly gift while I am on the phone.", "You can ask for time to review the organization and your budget. A legitimate charity should allow you to decide without pressure.", "Request the charity's full legal name and official website, then verify them independently. If you choose to give, use the organization's official payment channel."),
            ("I want to help a neighbor but cannot afford to give much.", "Support does not have to be large or monetary. Consider what you can offer without putting essential bills or your own safety at risk.", "Choose a clear amount or nonfinancial help that fits your circumstances. It is acceptable to set a boundary and revisit it later."),
            ("A disaster relief post asks for payment by cryptocurrency.", "Unusual or difficult-to-reverse payment methods can be a warning sign, especially when the organizer is unverified. Genuine needs can be imitated by scammers.", "Find established relief organizations through independent sources and donate through their verified websites if you decide to give."),
            ("Can I deduct a donation from my taxes?", "Tax deductions depend on the jurisdiction, recipient eligibility, documentation, and your personal tax circumstances. Not every donation qualifies.", "Check official tax-agency guidance or consult a qualified tax professional. Keep receipts and records, but do not assume a deduction before confirming eligibility."),
        ],
    ),
    (
        "Insurance as part of financial resilience",
        "Insurance transfers certain risks under a contract; it does not cover every loss. Compare premiums, deductibles, limits, exclusions, renewal terms, and the insurer's claims process. The lowest premium may mean less coverage or a higher deductible. Needs differ by household, local law, assets, and ability to absorb a loss.",
        "How do I compare two insurance policies?",
        "Compare coverage limits and exclusions alongside premiums and deductibles. Check whether the policies cover the risks you actually face, what documentation claims require, and whether prices can change at renewal. Ask the insurer to clarify ambiguous terms in writing.",
        [
            ("I am tempted to choose the policy with the lowest monthly premium.", "A low premium can come with higher deductibles, lower limits, or exclusions. Consider whether you could pay the deductible and whether the policy covers important risks.", "Put the premium, deductible, limits, and key exclusions side by side. Ask for written clarification before selecting coverage."),
            ("My employer provides life insurance. Is that enough?", "Employer coverage may be useful but can have limits or change when employment changes. Whether it is sufficient depends on dependents, debts, savings, and local products.", "Estimate what financial responsibilities would remain for people who depend on you. Review the policy terms and consider independent qualified advice if the decision is significant."),
            ("A salesperson says I need to buy an insurance add-on today.", "Pressure to decide immediately is a reason to slow down. Ask what risk it covers, what it costs over time, exclusions, cancellation rights, and whether you already have similar coverage.", "Request the policy document and compare it with existing coverage. Do not buy before understanding the terms and verifying the seller."),
            ("I filed a claim, but I am unsure what documents to send.", "Follow the insurer's written claim instructions and preserve copies of every submission. Do not send sensitive documents through an unverified channel.", "Contact the insurer through its official website or policy documents to confirm acceptable documents and deadlines. Keep a dated record of calls and claim references."),
        ],
    ),
    (
        "Retirement and long-term saving",
        "Long-term saving involves uncertainty, fees, tax rules, and access restrictions. Employer contributions, account types, investment choices, and withdrawal rules differ by country and plan. Read the plan documents and understand vesting, fees, and risks. A general example cannot determine the right contribution or investment mix for a person.",
        "I am young and retirement feels too far away to think about.",
        "Starting early can give contributions more time to grow, but current essentials and high-cost debt also matter. Learn the options available through your employer or local system, including fees, matching contributions, tax treatment, and withdrawal restrictions.",
        [
            ("My employer offers a contribution match, but I do not understand it.", "A match may be valuable compensation, but details such as eligibility, vesting, contribution limits, and investment choices matter.", "Read the plan summary and ask the benefits administrator how the match is calculated and when it becomes yours. Keep copies of the explanation."),
            ("The retirement account balance went down this month.", "Investment values fluctuate, and a short-term decline does not by itself show whether a long-term plan is appropriate. The right response depends on time horizon, risk tolerance, and portfolio details.", "Avoid making an impulsive change based on one statement. Review the investment allocation, fees, and plan documents, or seek qualified fiduciary advice where available."),
            ("I might need this retirement money for a house next year.", "Some retirement accounts restrict withdrawals or impose taxes, penalties, or loss of future growth. A near-term house goal may need a different type of savings.", "Read the account's withdrawal rules and local tax treatment before using it. Compare a separate, accessible savings plan for the house goal."),
            ("I do not know which fund to select in my workplace plan.", "Options may differ in fees, diversification, risk, and target date. Past returns do not guarantee future results, and one choice cannot be recommended without context.", "Read the plan's fund descriptions and fee disclosure. If available, use an independent qualified adviser who explains conflicts and costs."),
        ],
    ),
    (
        "Taxes and record keeping",
        "Tax rules depend on location, income, family circumstances, and the type of transaction. Keep organized records of income, deductible expenses where relevant, tax payments, and official notices. Use current official guidance and qualified professional help for complex situations. General educational examples should never substitute for jurisdiction-specific advice.",
        "What records should I keep for tax time?",
        "The records depend on local rules, but commonly include income statements, receipts relevant to deductions or credits, business records, tax payments, and prior filings. Keep them securely for the period required in your jurisdiction and avoid sending identity documents through unverified channels.",
        [
            ("I received a letter saying I owe additional tax and have a short deadline.", "Do not ignore the notice, but verify that it is genuine. Tax agencies have official contact methods, and scammers sometimes imitate notices or demand unusual payment.", "Use the agency's official website or a prior verified document to contact it. Note the response deadline and seek qualified local tax help if you do not understand the letter."),
            ("I do gig work and do not know how much to set aside for taxes.", "Tax obligations and deductible expenses vary by location and work type. Do not rely on a universal percentage as an exact answer.", "Track income and eligible expenses separately, consult official tax guidance for self-employment, and ask a qualified tax professional for advice specific to your situation."),
            ("I lost a receipt for a work-related purchase.", "Whether another record is acceptable depends on local tax rules and the type of expense. Never fabricate a receipt or claim an amount you cannot support.", "Check bank records, emails, and vendor account history for a duplicate. Consult official guidance or a qualified preparer about documentation requirements."),
            ("A friend says everyone should file taxes the same way to get a bigger refund.", "Tax filings depend on individual facts and local law. Incorrect claims can lead to repayment, penalties, or other consequences.", "Use official tax-agency resources or a qualified professional. Be cautious of preparers who promise a particular refund before reviewing documents."),
        ],
    ),
    (
        "Housing costs and lease decisions",
        "Housing affordability includes rent or mortgage payments, utilities, insurance, deposits, maintenance, transport, and moving costs. A landlord or lender's approval is not proof that a payment is comfortable. Read the contract, understand deposits and renewal terms, and verify local tenant protections. Housing law and customary costs vary widely.",
        "How much rent can I afford?",
        "Start with take-home income and all regular obligations, then include utilities, transport, insurance, savings, and variable costs. A common percentage guideline may not fit local prices or your household. Leave room for unexpected expenses and avoid basing the choice on gross income alone.",
        [
            ("The apartment is affordable only if I ignore utilities and commuting costs.", "Those costs are part of living there and can change the comparison with a different home. A low rent may not mean a low total housing cost.", "Estimate utilities from a reliable source and calculate commute expenses before signing. Compare the full monthly amount with your other obligations."),
            ("A landlord asks me to pay a deposit through a wire transfer before I view the place.", "Pressure to pay before verifying the property can be a rental scam warning. Rules for deposits vary by location, and payment methods can be difficult to reverse.", "Verify the landlord and property independently, view it or use a trusted representative, and check local rental rules before sending money. Do not share identity documents with an unverified contact."),
            ("My lease renewal increases the rent, and I need time to decide.", "Check the notice period, renewal deadline, and local rules. Compare the increase with moving costs and other available housing rather than looking at rent alone.", "Read the lease and contact a local tenant advice organization or official housing authority if you are unsure of your rights. Ask the landlord for written terms."),
            ("I am considering a mortgage because the payment looks close to my current rent.", "Mortgage ownership also includes taxes, insurance, repairs, fees, and possible rate changes. The upfront costs and financial risk differ from renting.", "Compare total monthly and one-time costs, ask about fixed versus variable terms, and obtain independent advice before committing. Keep a reserve for maintenance."),
        ],
    ),
    (
        "Education and training costs",
        "Education can improve opportunities but does not guarantee a particular job or income. Compare tuition, fees, books, equipment, lost work time, financing terms, completion rates, and realistic employment outcomes. Verify accreditation or recognition when relevant. Be cautious of high-pressure promises and loans whose total repayment you do not understand.",
        "Is an expensive career boot camp worth taking a loan for?",
        "The answer depends on verified outcomes, cost, financing, time, and alternatives. Ask for independently verifiable completion and placement data, check whether job claims are defined clearly, and compare lower-cost training or employer-sponsored options.",
        [
            ("The school promises a high salary after a short course.", "A marketing promise is not a guarantee of employment or income. Ask how the statistic was calculated, who was counted, and whether it is independently verified.", "Request the methodology and written refund and financing terms. Talk to former students independently and compare job postings with the skills taught."),
            ("I am deciding whether to borrow for a degree.", "Consider total cost, time out of work, grants, completion likelihood, likely field options, and repayment terms. Earnings estimates are uncertain and vary across individuals.", "Compare the full cost of multiple programs and ask the school about aid and completion outcomes. Use official loan disclosures and a trusted adviser before signing."),
            ("My employer may reimburse training after I pass.", "Reimbursement may have conditions, deadlines, repayment clauses, or a requirement to stay employed. Confirm the policy before paying upfront.", "Ask HR for the written rules, eligible providers, and what happens if employment changes. Avoid borrowing based only on an informal verbal promise."),
            ("I can study online for free, but I am unsure whether it will help me get hired.", "Free learning can build skills, while employers may also value demonstrated work, recognized credentials, or experience. Requirements vary by role.", "Review current job postings for your target role and list repeated requirements. Build a small portfolio or seek feedback before paying for a credential."),
        ],
    ),
    (
        "Family conversations about money",
        "Money conversations work best when they focus on shared facts, needs, and choices rather than blame. Agree on a time, protect private information, and distinguish shared bills from individual spending. Household arrangements differ; there is no single correct account structure. If a conversation feels unsafe or coercive, prioritize safety and seek trusted local support.",
        "How can we talk about money without arguing?",
        "Choose a calm time and one topic, such as an upcoming bill or shared goal. Use actual numbers, listen to each person's concern, and agree on a next review date. Avoid making a major financial decision while someone feels pressured or unable to ask questions.",
        [
            ("My partner hides purchases and we are arguing about trust.", "The issue may involve both spending and communication. Focus on specific facts and the impact on shared obligations rather than making assumptions about motive.", "Agree on what financial information must be shared for joint bills and goals. If secrecy involves coercion, threats, or control, consider confidential support from a trusted professional or local service."),
            ("My adult child asks me to take a loan for them.", "A loan in your name may make you legally responsible even if someone else promises to pay. Consider your ability to repay it yourself and what it would do to your own goals.", "Read the contract and avoid signing under pressure. You can offer help in a form you can afford, or decline without needing to take on debt."),
            ("My parents want help with expenses, but my siblings and I have different incomes.", "Fairness may mean different contributions rather than identical amounts. Clarify the actual need, time period, and each person's capacity without disclosing more private information than necessary.", "Discuss a specific shared plan, including what happens if the need changes. Consider nonfinancial support and local benefits or services too."),
            ("We share bills but keep separate bank accounts.", "Separate accounts can work when responsibilities and transfers are clear. Shared bills still need a reliable process and a plan for what happens if income timing changes.", "List shared expenses and agree on who pays each one, due dates, and how contributions are calculated. Review the arrangement after a major change."),
        ],
    ),
    (
        "Money boundaries and lending to friends",
        "Personal loans between friends can affect both finances and relationships. Before lending, consider whether you could afford never to get the money back, agree on written terms if appropriate, and avoid giving access to accounts. Saying no is allowed. Legal requirements and protections vary by location and amount.",
        "My friend asks to borrow money and promises to repay next week.",
        "A promise may be sincere, but repayment is uncertain. Decide based on your own ability to absorb the loss, not pressure or embarrassment. If you choose to help, be clear about amount, timing, and whether it is a loan or a gift.",
        [
            ("My friend has asked for money several times and has not repaid earlier loans.", "Repeated requests and unpaid balances increase the chance that another loan will not be repaid. You are allowed to set a limit even if the friend is in difficulty.", "Decide your boundary before replying and use a short, respectful statement. If you offer other help, choose something that does not put essential bills at risk."),
            ("Someone wants to borrow my bank login to receive a transfer.", "Do not share credentials or allow another person to use your account for an unexplained transfer. It could expose you to fraud, account loss, or legal problems.", "Decline and contact your bank if you already shared access. Never forward money from an unknown source without independently verifying the transaction and purpose."),
            ("I want to lend money but avoid awkwardness about repayment.", "Clear terms are kinder than vague expectations. Put the amount, repayment dates, and what happens if a payment is late in writing, where lawful and appropriate.", "Only lend an amount you can afford to lose. Consider whether a smaller gift or nonfinancial assistance would be healthier for the relationship."),
            ("A relative says I am selfish for not lending them money.", "A boundary is not proof that you do not care. Taking on debt or risking rent and essential expenses can harm both you and the relationship.", "Repeat your decision without debating every detail. If you want, offer a specific alternative such as helping them find a local support service."),
        ],
    ),
    (
        "Financial stress and small next steps",
        "Financial stress can make every decision feel urgent. A simple list of immediate essentials, due dates, balances, and available cash can restore some visibility. Start with one action, such as contacting a provider or setting a reminder. If debt, housing, food, or safety is at risk, seek reputable local services; support and protections vary by area.",
        "I am overwhelmed and do not know where to start.",
        "You do not need to solve everything today. Write down what is due next, what money is available, and which issue has the most serious consequence if ignored. Take one manageable step and ask for help from a trusted, qualified source when needed.",
        [
            ("I have overdue bills and keep avoiding the mail.", "Avoidance is understandable under stress, but unopened notices can contain deadlines or options. Open them with support if that makes the task manageable.", "Sort notices by due date and consequence, then contact one provider through its official channel to ask about options. Do not promise an amount before checking your budget."),
            ("I cannot afford both groceries and a utility bill this week.", "This is an immediate basic-needs problem, not a budgeting failure. Look for local food assistance and contact the utility about hardship protections or payment plans.", "Use official government or nonprofit directories for your location, and prioritize immediate safety. If someone in the household is at risk, contact an appropriate local support service."),
            ("I am embarrassed to ask a nonprofit counselor for help.", "Seeking help is a practical step, not a character judgment. Check the organization's credentials, fees, privacy practices, and whether it has conflicts of interest.", "Find a government or reputable nonprofit referral in your area and ask what the first appointment involves. Avoid organizations that guarantee a result or demand large upfront fees."),
            ("A debt company promises it can erase all my debt for a fee.", "Debt relief offers can involve risks, fees, credit consequences, and legal uncertainty. No company should guarantee an outcome without reviewing your situation.", "Read the contract carefully and check the company with relevant regulators or consumer agencies. Consider independent nonprofit counseling or local legal advice before stopping payments."),
        ],
    ),
    (
        "Protecting identity and documents",
        "Identity documents, tax records, account numbers, and recovery codes can be used to impersonate someone. Store records securely, share only what is required through verified channels, and use unique passwords with multi-factor authentication. If information is exposed, response steps depend on the country and institution; contact affected providers and official identity-protection resources promptly.",
        "A rental application asks for my identity documents through a messaging app.",
        "Before sending sensitive documents, verify the landlord, property, application process, and secure submission method. Ask why each document is needed and whether less sensitive proof is acceptable. Local privacy rules vary.",
        [
            ("I received an email attachment claiming to be a tax refund form.", "Unexpected attachments can contain malware or phishing requests. Do not open it or enter credentials through a linked page until you verify the sender.", "Contact the tax agency through its official website and report the message. If opened, use trusted security software and seek local technical help if suspicious activity follows."),
            ("I lost my wallet with cards and identity documents.", "Act quickly to reduce account and identity risks. Contact card issuers through official channels, monitor accounts, and follow local procedures for replacing documents or reporting loss.", "Make a list of what was inside, freeze or cancel payment cards as appropriate, and check official government identity-protection guidance for your area."),
            ("A caller already knows my address and birth date, so I assume they are legitimate.", "Personal details can be obtained from many sources and do not authenticate a caller. Do not share passwords, codes, or additional sensitive information.", "End the call and contact the organization independently. Review account alerts and change credentials if you disclosed any authentication information."),
            ("I use the same password for banking and shopping sites.", "If one site is compromised, reused passwords can put other accounts at risk. Unique passwords reduce this chain of exposure.", "Change the most sensitive passwords from a trusted device and use a reputable password manager if it suits you. Enable multi-factor authentication where available."),
        ],
    ),
    (
        "Online marketplaces and payment safety",
        "Peer-to-peer sales and payment apps may offer limited recovery if a transaction goes wrong. Verify the buyer, seller, item, and payment method; understand platform protections; and avoid moving a transaction off-platform under pressure. Never share one-time codes. Local laws and the platform's terms determine available remedies.",
        "How can I complete a marketplace transaction safely?",
        "Check the platform's current protections and keep communication and payment on-platform when required. Verify payment inside the official app, inspect the item or shipping details, and save receipts and messages. Do not share passwords or one-time codes, and remember that some transfers cannot be reversed.",
        [
            ("A buyer wants to overpay and asks me to refund the difference.", "An overpayment followed by a refund request is a common fraud pattern. The original payment may be reversed after you send money, leaving you responsible for the loss.", "Do not send a refund outside the platform. Verify the payment in your account directly, contact platform support through its official app, and follow its instructions."),
            ("A seller says I must pay a deposit before seeing an expensive item.", "An advance payment can be difficult to recover if the seller or listing is fake. Platform protections may not cover off-platform transfers.", "Verify the listing and seller, use a protected payment method permitted by the platform, and do not send money under pressure."),
            ("Someone asks me to share a code to prove I am a real seller.", "A one-time code may grant account access or authorize a transaction. It is not a safe identity check to share with another person.", "Do not share it. Secure the account and contact the platform through its official support channel if you already revealed the code."),
            ("The platform says the item was delivered, but I never received it.", "Save order details and tracking information, check with household members or the carrier, and use the platform's dispute process before the deadline.", "Contact the carrier through its official tracking service and open a case with platform support. Keep screenshots and avoid moving the dispute to private messages."),
            ("A stranger wants to pay me by sending a link to 'release' my funds.", "A payment link may be designed to steal credentials or card details. Funds should be verified in the official app, not through a link from an unknown person.", "Do not open the link or enter account details. Check the transaction by opening the service directly and report suspicious contact."),
        ],
    ),
    (
        "Inflation and changing prices",
        "Inflation means prices rise over time on average, but each household's experience depends on what it buys. Compare actual prices for essential categories and adjust the plan based on evidence. Do not make rushed investment or borrowing decisions from a headline. Rates, wages, and prices change differently, and future outcomes are uncertain.",
        "My grocery bill keeps rising even though I buy the same things.",
        "Track the unit price and quantity for a few regular items. Package size, store, season, and promotions can change the real cost. Compare alternatives while considering nutrition, transport, and time rather than assuming the cheapest sticker price is always best.",
        [
            ("I am switching brands to save money but worry about quality.", "Compare ingredients, size, and the purpose of the product. A cheaper choice is only useful if it meets your needs and is safe.", "Try a small amount before buying in bulk and compare unit prices. Keep essential dietary or health needs in mind."),
            ("My rent and utilities increased at the same time.", "Housing and utility changes can overwhelm a budget because they are large recurring costs. Check the notices, contract, and local rules rather than assuming every increase is correct.", "Contact the provider for an itemized explanation and look for official tenant or utility assistance resources in your area."),
            ("Should I borrow now because prices may rise further?", "Borrowing to anticipate future prices adds interest and repayment risk. A forecast may be wrong, and the purchase may not be necessary.", "Separate a genuine near-term need from a fear of missing out. Compare the cash price and total loan cost, and avoid debt you cannot repay under a conservative budget."),
            ("Does inflation mean I should invest all my savings?", "No investment guarantees protection from inflation, and market values can fall. Emergency and near-term money may need stability and access more than growth.", "Match each savings goal to its time horizon and risk capacity. Learn fees and local protections before choosing financial products."),
        ],
    ),
    (
        "Setting up useful reminders",
        "A reminder system can prevent missed bills and reduce mental load. Use calendar alerts, account notifications, or a paper checklist, but protect private details on shared devices. Confirm that alerts are enabled and understand whether a payment is scheduled or merely reminded. Technology helps organize decisions but does not replace checking balances and statements.",
        "I keep forgetting due dates even though I have enough money.",
        "Put each due date in one trusted calendar and set an alert early enough to act. If using automatic payments, verify the amount and account balance before the date; automatic payments can still fail or change.",
        [
            ("I get too many bank notifications and ignore them all.", "Too many alerts can make important ones easy to miss. Choose a small set for low balances, large transactions, payment due dates, and account changes.", "Review notification settings in the official app and remove alerts that do not help. Keep security alerts enabled when possible and protect access to the device."),
            ("My income arrives after a bill's due date.", "The timing mismatch can cause fees even when income is sufficient overall. Some providers may offer another due date, but policies vary.", "Contact the provider through an official channel and ask whether the due date can be changed. Confirm any agreement in writing and do not assume it applies until approved."),
            ("I set up autopay but the amount changes every month.", "Variable bills need a review process because automatic payment may withdraw a higher amount than expected. Alerts or payment caps may be available depending on the provider.", "Check whether the provider offers a notification before payment. Review each statement and keep an adequate balance; do not enable a feature without understanding its terms."),
            ("My phone is shared with family, and financial notifications appear on screen.", "Lock-screen notifications can reveal private balances or transactions. Privacy settings differ by device and app.", "Hide sensitive notification previews and use a strong device passcode. Sign out of financial accounts on devices you do not control."),
        ],
    ),
    (
        "Choosing financial information sources",
        "Financial information online can mix education, advertising, and personal opinion. Check the author's qualifications, date, evidence, conflicts, and whether the rules apply in your jurisdiction. Be cautious of guaranteed outcomes, referral commissions that are not disclosed, and pressure to act immediately. For consequential decisions, verify facts through official or independently qualified sources.",
        "A popular video says a new investment method cannot lose. Should I trust it?",
        "No investment method can be assumed risk-free based on a video. Check what the creator sells, whether compensation is disclosed, how losses are described, and whether independent evidence supports the claims.",
        [
            ("A financial influencer says everyone should buy the same stock.", "A recommendation may not fit your goals or risk capacity, and the speaker may have incentives or undisclosed holdings. Concentrated investments can lose value.", "Do not act solely on a social post. Research official filings and independent sources, understand the risk, and seek qualified advice if needed."),
            ("An article is five years old but appears near the top of search results.", "Rates, tax rules, products, and laws can change. An older article may still teach a concept but not current details.", "Check the publication and update dates, then verify current rules through official sources for your location."),
            ("A company offers a free financial consultation but asks me to sign up for a product.", "The consultation may be a sales conversation. Ask how the representative is paid and whether they have a fiduciary duty or other conflicts under local rules.", "Request the fee and compensation information in writing. Take time to compare other sources and do not sign anything you do not understand."),
            ("Two reputable sources give different advice.", "Different recommendations may reflect different assumptions, goals, or local rules. Check what facts each source assumes and whether those assumptions match your situation.", "Write down the decision and the uncertainties, then consult an independent qualified professional for a high-stakes choice. Avoid treating disagreement as proof that one source is dishonest."),
        ],
    ),
    (
        "Starting a small business budget",
        "Business cash flow is different from profit: invoices may be paid late, taxes and operating costs may be due later, and personal spending still needs a plan. Keep records, separate accounts where practical, and understand local registration and tax requirements. Forecast conservatively and do not spend expected revenue before it is received.",
        "My small business made sales this month. How much can I pay myself?",
        "Sales are not the same as available personal income. First account for refunds, costs, taxes, payroll or contractor obligations, debt, and upcoming bills. Local legal structures affect how owner payments should be handled.",
        [
            ("A customer has not paid an invoice, but I need to cover rent.", "An unpaid invoice is not cash in hand. Review the contract, due date, and lawful follow-up options, and plan for a delay in future cash-flow estimates.", "Contact the customer through documented business channels and ask about payment status. Avoid borrowing against an uncertain payment without comparing fees and repayment risk."),
            ("I mix business purchases with personal card spending.", "Mixed records make it harder to understand cash flow and may complicate tax or legal reporting. Requirements vary by business structure and location.", "Start categorizing transactions and open a separate business account if appropriate and affordable. Ask a qualified local accountant how to handle past mixed expenses."),
            ("A vendor offers a discount if I pay for a year of software upfront.", "Compare the discount with cash-flow needs, renewal terms, cancellation rules, and whether the business is likely to use the service for the full year.", "Calculate the total savings and the cash you would no longer have for operations. Read renewal terms and set a reminder before the next billing date."),
            ("I want to hire someone, but revenue has been inconsistent.", "Hiring adds ongoing obligations that can continue during a slow period. Consider workload, legal classification, taxes, insurance, and cash reserves before committing.", "Estimate total employment cost and compare it with conservative revenue scenarios. Check local labor rules and consult a qualified professional before making an offer."),
        ],
    ),
    (
        "Preparing for a move",
        "Moving can involve deposits, application fees, travel, utility setup, storage, cleaning, and temporary overlap between homes. A move budget should include one-time costs and changes to recurring expenses. Keep receipts and understand refund rules. Verify new housing and payment instructions to reduce rental fraud risk.",
        "I found a place that fits my rent budget. What else should I calculate?",
        "Include utilities, commute, deposit, moving services, application costs, parking, internet, renter's insurance, and any overlap with current housing. Confirm which expenses are refundable and when they are due.",
        [
            ("I need to pay a deposit before my next paycheck.", "A deposit can strain cash flow, and its legal treatment varies by location. Avoid high-cost borrowing without understanding the repayment and alternatives.", "Ask the landlord about lawful payment timing and verify the property first. Check local tenant resources for deposit limits and documentation requirements."),
            ("A moving company quote is much lower than others.", "A low quote may exclude services or change after an in-home estimate. Verify the company's identity, insurance, written scope, and cancellation policy.", "Request a detailed written quote and check independent licensing or consumer records where available. Do not pay a large unexplained fee before verifying the provider."),
            ("I will have to pay rent in two places for one month.", "Temporary overlap is a real one-time housing cost. Include it in the move plan and avoid treating the new monthly rent as the only affordability number.", "Confirm lease dates and notice requirements in writing. Reserve the overlap amount before spending on optional moving upgrades."),
            ("My new landlord says I should wire the deposit today to secure the unit.", "Urgency and irreversible payment before verifying the property are warning signs. Legitimate processes should be documented and consistent with local rules.", "Independently verify ownership or management, inspect the unit or use a trusted representative, and read the lease before sending money."),
        ],
    ),
    (
        "Making a spending plan for holidays",
        "Holiday costs often include gifts, travel, food, events, and time away from work. Decide on an overall amount before shopping and make smaller categories within it. Agree on gift expectations with family when possible. A celebration does not need to create debt; lower-cost traditions can be meaningful.",
        "I want to enjoy the holidays without carrying credit-card debt into the new year.",
        "Set a total amount based on money available after essentials and required payments, then track purchases against it. Include travel and meals, not only gifts. If the total is too high, reduce or simplify plans before spending.",
        [
            ("My family expects expensive gifts.", "You can set a limit and suggest a shared price cap, exchange, or experience instead. Expectations can be discussed without criticizing anyone else's choices.", "Decide your budget first and communicate it early. A sincere explanation can be brief; you do not need to borrow to prove care."),
            ("I have travel plans but have not priced food or local transport.", "The ticket is only one part of a trip. Include lodging, meals, local transit, baggage, insurance, and possible schedule changes.", "Make a full itinerary estimate and add a reasonable contingency if possible. Compare the total with your savings before purchasing nonrefundable travel."),
            ("A buy-now-pay-later offer makes a gift seem affordable.", "Installments still create obligations, and multiple plans can overlap. Check fees, late consequences, payment dates, and total cost.", "Add every installment to your bill calendar before deciding. If the purchase does not fit without uncertain future income, choose a lower-cost gift."),
            ("I overspent last holiday and feel bad about it.", "Use the experience as information rather than a reason for shame. Identify which costs were predictable and which decisions were rushed.", "Write a modest plan for the next holiday and set aside a small amount periodically if possible. You can also choose new traditions that cost less."),
        ],
    ),
    (
        "Financial choices after a life change",
        "A new job, separation, child, move, illness, or loss can change income, benefits, accounts, and legal responsibilities. Avoid rushed irreversible decisions when possible. Update a cash-flow plan, review beneficiaries and documents where appropriate, and get qualified legal, tax, or financial advice for complex matters. Rules and rights depend on location.",
        "I am changing jobs and will have a gap between paychecks.",
        "Estimate the final paycheck, new start date, benefits transition, and essential costs during the gap. Confirm when health coverage and other benefits begin, and do not assume old benefits continue automatically.",
        [
            ("I am leaving a job and need to decide what to do with a retirement account.", "Options, fees, tax treatment, and deadlines depend on the plan and jurisdiction. Avoid moving funds based only on a sales pitch.", "Read the plan's written options and consult a qualified independent adviser or official retirement resource. Keep records and confirm any tax consequences before acting."),
            ("A separation means we need to divide household bills quickly.", "First protect immediate housing, utilities, and access to essential accounts. Legal ownership and obligations depend on agreements and local law.", "Make a list of accounts, due dates, and whose name is on each contract. Seek local legal or nonprofit support before changing joint accounts or signing agreements."),
            ("I am expecting a child and want to plan for the added costs.", "Costs may include medical care, leave, childcare, supplies, insurance, and changes to income. Benefits and protections vary by employer and location.", "Ask the employer and official local agencies about leave and benefits, then estimate recurring and one-time costs. Start with essential needs rather than buying every advertised product."),
            ("A family member died and I have been asked to handle their finances.", "This can involve legal authority, estate rules, creditors, and sensitive identity information. Do not transfer or distribute assets without confirming your authority.", "Contact the relevant institution and local probate or legal resource using official channels. Keep records and seek qualified professional help for complex matters."),
        ],
    ),
]


def build_dialogues():
    sections = []
    for title, guidance, followup, followup_answer, scenarios in TOPICS:
        for scenario, question, next_step in scenarios:
            sections.append(
                f"Human: {scenario} {question}\n"
                f"Assistant: {guidance}\n"
                f"In this situation, {next_step}\n"
                f"Human: {followup}\n"
                f"Assistant: {followup_answer}\n"
                f"Remember: {title} is a general learning topic. Verify current terms, "
                "local rules, and your own circumstances before making a consequential decision.\n"
            )
    return MARKER + "\n".join(sections)


def main():
    existing = OUTPUT.read_text(encoding="utf-8")
    base = existing.split(MARKER, maxsplit=1)[0].rstrip()
    expanded = base + "\n" + build_dialogues()
    OUTPUT.write_text(expanded, encoding="utf-8", newline="\n")
    size = OUTPUT.stat().st_size
    print(f"Wrote {OUTPUT} ({size:,} bytes; {len(TOPICS)} topics).")
    if not 150_000 <= size <= 250_000:
        raise SystemExit(
            "The chat corpus is outside the requested 150–250 KB range; "
            "adjust the scenario content before using it."
        )


if __name__ == "__main__":
    main()
